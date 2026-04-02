# 06_LoRA 微调训练

本章对应 `minimind_src/trainer/train_lora.py`。目标是在不更新原始大部分参数的前提下，仅训练 LoRA 注入到 `nn.Linear` 上的低秩增量，从而完成对话/指令 SFT 风格数据的微调。

## 1. LoRA 的核心理论支撑 (L14)

在深入代码前，需掌握 LoRA (Low-Rank Adaptation) 的两个关键假设：
- **低秩假设**：模型在适配下游任务时，权重的变化量 $\Delta W$ 实际上是低秩的（Intrinsic Rank 远小于 $d$）。
- **低秩分解**：将 $\Delta W$ 分解为两个小矩阵的乘积：$\Delta W = B \times A$，其中 $A \in \mathbb{R}^{r \times d}$，$B \in \mathbb{R}^{d \times r}$，$r \ll d$。参数量从 $d^2$ 压缩至 $2dr$。
- **训练优势**：
  - **显存友好**：由于主模型冻结，优化器状态（动量等）仅需为 2% 的参数维护。
  - **收敛快速**：在更小的参数空间内搜索，通常比全参微调更早达到性能平原。
  - **无损推理**：权重合并后，推理延迟与原模型完全一致。

本章你需要重点掌握的链路是：

- `init_model(...)`：先加载“基底 dense 权重”（通常来自 `full_sft` 或 `pretrain`）
- `apply_lora(model)`：给模型所有目标 `Linear` 注入 `module.lora` 并替换 forward 为“原输出 + LoRA 输出”
- 冻结非 LoRA 参数：只留下包含 `'lora'` 的参数 `requires_grad=True`
- optimizer 只管理 LoRA 参数：`AdamW(lora_params, lr=...)`
- 保存时：
  - 每次保存权重快照：调用 `save_lora(model, lora_save_path)`，只保存 LoRA 子模块权重
  - 同时调用 `lm_checkpoint(..., model=model, optimizer=..., scaler=...)` 保存可 resume 的完整训练状态

---

## 1. Import（逐行）

```python
1: import os
2: import sys
...
7: import argparse
8: import time
9: import warnings
10: import torch
11: import torch.distributed as dist
12: from contextlib import nullcontext
13: from torch import optim, nn
14: from torch.nn.parallel import DistributedDataParallel
15: from torch.utils.data import DataLoader, DistributedSampler
16: from model.model_minimind import MiniMindConfig
17: from dataset.lm_dataset import SFTDataset
18: from model.model_lora import save_lora, apply_lora
19: from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
```

- `apply_lora/save_lora`：本章的核心是利用 `model_lora.py` 提供的注入与保存逻辑。
- `SFTDataset`：LoRA 微调使用 SFT 数据集，labels 语义来自 `03` 章。
- `lm_checkpoint`：用于保存 optimizer/scaler 等 resume 状态（见 `04` 章）。
- `DistributedSampler/DistributedDataParallel`：支持多卡。

---

## 2. `train_epoch`：LoRA-only 训练步（逐行+张量语义）

### 2.1 函数签名与参数含义

```python
24: def train_epoch(epoch, loader, iters, lora_params, start_step=0, wandb=None):
```

- `lora_params`：一个由 LoRA 参数 tensor 组成的 list，仅这些参数会 `requires_grad=True` 且被 optimizer 管理。

### 2.2 DataLoader batch 进入训练

```python
27: for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
28:     input_ids = input_ids.to(args.device)
29:     labels = labels.to(args.device)
30:     last_step = step
```

张量维度（batch 内）：

- 来自 `SFTDataset.__getitem__` 的单样本 `[T]`：
  - batch 后变为 `input_ids: [B, T]`
  - 同理 `labels: [B, T]`

### 2.3 学习率更新

```python
31: lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
32: for param_group in optimizer.param_groups:
33:     param_group['lr'] = lr
```

- `get_lr(...)` 返回标量；写入每个 optimizer param_group。

### 2.4 forward 与 loss 合成（aux_loss 仍然参与反传）

```python
35: with autocast_ctx:
36:     res = model(input_ids, labels=labels)
37:     loss = res.loss + res.aux_loss
38:     loss = loss / args.accumulation_steps
```

关键点：

- `res.loss` 来自 `MiniMindForCausalLM` 的 cross entropy shift（`labels` 为 `-100` 的位置被 ignore）。
- `res.aux_loss` 来自 MoE router 正则项：
  - dense 模式下通常是一个 0 标量（或 `aux_loss` 为 0）。
  - MoE 模式下 `aux_loss` 为标量，仍按同样方式参与反传。
- `loss / accumulation_steps`：梯度累积时保持有效步长一致。

### 2.5 AMP backward / clip / step / zero_grad

```python
39: scaler.scale(loss).backward()
41: if step % args.accumulation_steps == 0:
42:     scaler.unscale_(optimizer)
43:     torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
44:     scaler.step(optimizer)
45:     scaler.update()
46:     optimizer.zero_grad(set_to_none=True)
```

精确含义：

- `scaler.scale(loss)`：放大梯度以适配 fp16/bf16 的数值范围。
- `unscale_`：在 clip 前取消缩放，保证 clip 的阈值单位正确。
- `clip_grad_norm_(lora_params, ...)`：
  - 这里 clip 只作用在 `lora_params`，与 optimizer 管理参数一致
  - 因而不会浪费时间在冻结参数上进行梯度计算（被冻结的参数理论上不会有梯度，或 grad 为 None）。

### 2.6 日志与保存（逐行）

日志：

```python
49: if step % args.log_interval == 0 or step == iters:
50:     current_loss = loss.item() * args.accumulation_steps
51:     current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
53:     current_logits_loss = current_loss - current_aux_loss
...
```

这里 `loss.item()` 是“除以 accumulation_steps 之后的 loss 标量”，乘回 `args.accumulation_steps` 得到未除的标量，便于和其它日志对齐。

保存权重快照与 resume：

```python
59: if (step % args.save_interval == 0 or step == iters) and is_main_process():
60:     model.eval()
61:     lora_save_path = f'{args.save_dir}/{args.lora_name}_{lm_config.hidden_size}.pth'
62:     save_lora(model, lora_save_path)
64:     lm_checkpoint(lm_config, weight=args.lora_name, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
65:     model.train()
```

保存的两类文件含义（与 `04` 章相互对应）：

1. `lora_save_path/*.pth`：
   - 由 `save_lora` 保存的 state_dict，只包含带 `.lora.` 的权重键。
   - 这是“只保存 LoRA 权重”的轻量快照。
2. `../checkpoints/*_resume.pth`（由 `lm_checkpoint` 产生）：
   - 保存了 optimizer/scaler 等训练可恢复状态（训练 resume 需要它）。

最后的 step 边界补更新：

```python
69: if last_step > start_step and last_step % args.accumulation_steps != 0:
70:     scaler.unscale_(optimizer)
71:     clip_grad_norm_(lora_params, args.grad_clip)
72:     scaler.step(optimizer)
73:     scaler.update()
74:     optimizer.zero_grad(set_to_none=True)
```

---

## 3. `__main__`：LoRA 训练启动流程（逐行）

### 3.1 argparse 参数含义

本脚本的关键 args：

- `--lora_name`：LoRA 权重命名前缀，也用于 checkpoint 文件名
- `--from_weight`：基底 dense 权重（通常 `full_sft`）
- `--from_resume`：是否从之前的 `*_resume.pth` 续训
- `--use_moe`：决定模型结构是否为 MoE（从而影响 `aux_loss` 是否为非零）

### 3.2 DDP 初始化与 seed

```python
103: local_rank = init_distributed_mode()
104: if dist.is_initialized(): args.device = f"cuda:{local_rank}"
105: setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

- 单卡：`init_distributed_mode` 返回 0 且 dist 未初始化。
- 多卡：seed 基于 rank 变化，保证数据与 dropout 等在各卡之间不同。

### 3.3 config 与（可选）resume checkpoint 读取

```python
108: os.makedirs(args.save_dir, exist_ok=True)
109: lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
110: ckp_data = lm_checkpoint(lm_config, weight=args.lora_name, save_dir='../checkpoints') if args.from_resume==1 else None
```

- 只有当 `from_resume==1` 才加载 `ckp_data`。
- `lm_checkpoint` 在加载模式会找 `resume_path` 并返回 ckp_data 或 None。

### 3.4 AMP autocast 上下文与 wandb

```python
113: device_type = "cuda" if "cuda" in args.device else "cpu"
114: dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
115: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
...
if args.use_wandb and is_main_process(): ... wandb.init(...)
```

### 3.5 init_model -> apply_lora -> 冻结非 LoRA 参数（核心逻辑逐行）

```python
127: model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
128: apply_lora(model)
```

这里的顺序很关键：

- `init_model` 在 `apply_lora` 之前创建 `MiniMindForCausalLM` 并加载基底 dense 权重；
- `apply_lora` 之后才在目标 `Linear` 上挂上 `module.lora` 并替换 forward。

统计与冻结：

```python
130: total_params = sum(p.numel() for p in model.parameters())
131: lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)
137: lora_params = []
138: for name, param in model.named_parameters():
139:     if 'lora' in name:
140:         param.requires_grad = True
141:         lora_params.append(param)
142:     else:
143:         param.requires_grad = False
```

精确含义：

- 通过名字包含 `'lora'` 来判断是否属于 LoRA 参数（与 `save_lora/load_lora` 的 key 命名风格一致）。
- 结果保证：
  - 非 LoRA 参数梯度不会参与优化（requires_grad=False）
  - optimizer 的参数集合仅包含 LoRA 参数列表 `lora_params`

### 3.6 DataLoader / optimizer / scaler

```python
147: train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
148: train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
149: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
150: optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
```

维度与 dtype：

- SFTDataset 输出 `[T]`，batch 后 `[B,T]`。
- scaler 仅在 `args.dtype=='float16'` 时启用；bf16 通常不需要动态 loss scaling。

### 3.7 resume：加载 model/optimizer/scaler 状态（逐行）

```python
152: start_epoch, start_step = 0, 0
154: if ckp_data:
155:     model.load_state_dict(ckp_data['model'], strict=False)
156:     optimizer.load_state_dict(ckp_data['optimizer'])
157:     scaler.load_state_dict(ckp_data['scaler'])
158:     start_epoch = ckp_data['epoch']
159:     start_step = ckp_data.get('step', 0)
```

注意：

- `strict=False`：兼容 MoE/dense 或 LoRA key 变化造成的部分缺失。
- 由于此时已经 `apply_lora(model)`，所以 resume 的 `model` state_dict 若包含 `.lora.` 权重，就能正确装载。

### 3.8 torch.compile 与 DDP wrapper

```python
162: if args.use_compile == 1:
163:     model = torch.compile(model)
164:     Logger('torch.compile enabled')
165: if dist.is_initialized():
166:     model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
167:     model = DistributedDataParallel(model, device_ids=[local_rank])
```

忽略 buffer：

- `freqs_cos/freqs_sin` 是 position buffer，跳过 DDP 可能减少不必要的同步。

### 3.9 每 epoch 的 skip 续训与 DataLoader 构造

```python
170: for epoch in range(start_epoch, args.epochs):
171:     train_sampler and train_sampler.set_epoch(epoch)
172:     setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
173:     skip = start_step if (epoch == start_epoch and start_step > 0) else 0
174:     batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
175:     loader = DataLoader(train_ds, batch_sampler=batch_sampler, ...)
176:     if skip > 0:
177:         train_epoch(..., len(loader) + skip, lora_params, start_step, wandb)
178:     else:
180:         train_epoch(..., len(loader), lora_params, 0, wandb)
```

这与 `04` 章的 `SkipBatchSampler` 语义严格对齐。

---

## 4. 本章小结（你应该能复核的检查点）

1. 注入与 forward：
   - `apply_lora(model)` 通过替换每个目标 `Linear.forward` 实现 `Linear(x) + LoRA(x)`。
2. 冻结策略：
   - 仅 `'lora'` 命名空间下的参数 `requires_grad=True`，其它参数全部冻结。
3. optimizer 参数集合：
   - `optimizer = AdamW(lora_params, ...)` 严格只管理 LoRA 参数。
4. 保存策略：
   - `save_lora(model, lora_save_path)` 只保存 `.lora.` 权重键（轻量快照）
   - `lm_checkpoint(..., model=model, optimizer=..., scaler=...)` 保存 resume 状态
5. 梯度累积：
   - `loss = (res.loss + res.aux_loss) / accumulation_steps`
   - 只在整除点执行 unscale/clip/step/update。

---

## 5. 面试高频考点 (L14)

### Q1: 为什么 LoRA 训练中主模型不更新，却仍然占用大量显存？
答：虽然主模型权重被冻结（不需要存梯度和优化器状态），但**前向传播的激活值 (Activations)** 仍然需要保存在显存中，以便在反向传播时计算 LoRA 旁路矩阵 $A$ 和 $B$ 的梯度。

### Q2: LoRA 的 $B$ 矩阵为什么必须初始化为全 0？
答：确保训练开始时 $\Delta W = B \times A = 0$。这使得模型初始输出与原预训练模型完全一致，保证了微调的起点是稳健的，不会因为随机初始化的旁路而破坏预训练知识。

### Q3: 既然 LoRA 效果好，为什么不直接训练一个 $r \times r$ 的小模型？
答：LoRA 的有效性建立在预训练模型学到的**高维特征空间**之上。低秩增量是对这一成熟空间的微调，而非在低维空间重新学习知识。

### Q4: 训练中如何选择秩 $r$？
答：通常 $r=8$ 或 $16$ 已能满足大部分指令遵循任务。秩越大，模型表达能力越强，但显存占用增加且过拟合风险变大。MiniMind 默认使用 $r=16$。

下一章（`07_DPO（偏好对齐）`）我会逐行解析 `train_dpo.py`：从 `DPODataset` 输出的 `x/y/mask` 到 `dpo_loss` 里 chosen/rejected 的切分逻辑与 mask 聚合方式，解释为什么 DPO 的损失能落在 logprob 差值上，而不是直接 CE。
