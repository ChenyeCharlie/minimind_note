# 05_预训练与全量 SFT 训练循环

本章覆盖两份训练入口脚本：

- `minimind_src/trainer/train_pretrain.py`（预训练 / next-token 语言建模）
- `minimind_src/trainer/train_full_sft.py`（全量 SFT / 对话监督）

两者在工程结构上几乎完全相同，本章会把“共同的训练循环机制”讲到逐行可复核，同时在关键差异处强调：

- 数据集不同（`PretrainDataset` vs `SFTDataset`）
- labels 的语义不同（预训练：`labels=input_ids`；SFT：只有 assistant 段是有效 label，其他位置 `-100`）
- 训练 loss 的含义相同接口：`res.loss + res.aux_loss`

---

## 1. 共同张量约定（用于本章后续所有维度解释）

在训练脚本中，DataLoader 的 batch 一般来自 Dataset 的 `__getitem__` 返回值：

- 预训练：`(input_ids, labels)`，其中单样本为 `[T]`（`T=max_seq_len`），batch 后为：
  - `input_ids: [B, T]`
  - `labels: [B, T]`
- SFT：`(input_ids, labels)` 同样是
  - `input_ids: [B, T]`
  - `labels: [B, T]`

模型 forward（见 `MiniMindForCausalLM.forward`）在内部做 self-model forward，再做：

- logits：`[B, T, V]`
- loss shift：`x=logits[..., :-1, :]` 与 `y=labels[..., 1:]` 对齐
  - `x: [B, T-1, V]`
  - `y: [B, T-1]`

因此训练脚本里即使只传 `(input_ids, labels)`，loss 的对齐维度会由模型实现保证。

---

## 2. 预训练入口：`train_pretrain.py`

### 2.1 import 与依赖（逐行）

```python
1: import os
2: import sys
4: __package__ = "trainer"
5: sys.path.append(...)
7: import argparse
8: import time
...
16: from model.model_minimind import MiniMindConfig
17: from dataset.lm_dataset import PretrainDataset
18: from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
```

说明：

- `PretrainDataset` 在本章提供 `labels`（pad -> `-100`）。
- `init_model` 负责加载 tokenizer 和初始化 `MiniMindForCausalLM`（并加载 `from_weight` 权重）。
- `lm_checkpoint` 负责保存/恢复训练状态（完整 resume）。

---

### 2.2 `train_epoch`：逐行解析训练步（含 AMP 与梯度累积）

#### 2.2.1 训练循环骨架

```python
23: def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
26:     for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
27:         input_ids = input_ids.to(args.device)
28:         labels = labels.to(args.device)
29:         last_step = step
```

维度：

- loader batch：`input_ids: [B,T]`，`labels: [B,T]`（单样本 `[T]` 经 batch 叠加得到 `[B,T]`）
- `.to(args.device)` 不改变维度，只改变 dtype/device。

#### 2.2.2 学习率手动更新

```python
30: lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
31: for param_group in optimizer.param_groups:
32:     param_group['lr'] = lr
```

- `get_lr(...)` 返回标量。
- AdamW 可能有多个 param_group（例如未来可做分组 lr），这里无条件写入所有 group。

#### 2.2.3 AMP autocast + forward + loss 合成

```python
34: with autocast_ctx:
35:     res = model(input_ids, labels=labels)
36:     loss = res.loss + res.aux_loss
37:     loss = loss / args.accumulation_steps
```

逐行含义：

- `res = model(..., labels=labels)`：
  - `MiniMindForCausalLM.forward` 返回 `loss`（来自 cross_entropy shift）
  - 同时返回 `aux_loss`（来自 MoE router 正则项；dense 时通常是 0 标量）
- `loss = res.loss + res.aux_loss`：
  - 标量相加，不改变维度。
- `loss = loss / args.accumulation_steps`：
  - 目的：梯度累积时让“有效 batch 的 loss 尺度”保持一致，等价于对累积的多个 mini-batch 求和后的平均。

维度总结：

- `loss`：标量（0-dim tensor）。

#### 2.2.4 backward（由 GradScaler 控制的缩放）

```python
39: scaler.scale(loss).backward()
```

- 这是 AMP 的典型流程：
  - `scaler.scale(loss)` 通过放大梯度避免 fp16 underflow。

#### 2.2.5 梯度累积边界：unscale、clip、step

```python
41: if step % args.accumulation_steps == 0:
42:     scaler.unscale_(optimizer)
43:     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
44:     scaler.step(optimizer)
45:     scaler.update()
46:     optimizer.zero_grad(set_to_none=True)
```

逐行：

- 条件 `step % accumulation_steps == 0`：
  - 表示累计了若干个 backward 后才真正更新参数。
- `unscale_`：
  - 取消 scale 以便正确做 `clip_grad_norm_`。
- `clip_grad_norm_`：
  - clip 范数，稳定训练，避免梯度爆炸。
- `scaler.step(optimizer)`：
  - 根据 scaler 状态执行 optimizer step。
- `scaler.update()` 更新动态 loss scaling。
- `zero_grad(set_to_none=True)`：
  - 清空梯度，`set_to_none=True` 更省内存。

#### 2.2.6 日志打印的 loss / aux_loss 拆分（标量）

```python
50: if step % args.log_interval == 0 or step == iters:
51:     current_loss = loss.item() * args.accumulation_steps
52:     current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
53:     current_logits_loss = current_loss - current_aux_loss
...
57:     Logger(f'Epoch:[...], loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, ...')
```

说明：

- 注意这里用 `loss.item() * accumulation_steps`：
  - 因为前面把 loss 除过 `accumulation_steps`，这里恢复成“未除的标量 loss 值”做日志。
- `current_logits_loss = current_loss - current_aux_loss`：
  - 即 logit/CE 项分量（近似）。

#### 2.2.7 保存 checkpoint（两层：权重文件 + resume pth）

```python
60: if (step % args.save_interval == 0 or step == iters) and is_main_process():
61:     model.eval()
62:     moe_suffix = '_moe' if lm_config.use_moe else ''
63:     ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
64:     raw_model = model.module if isinstance(model, DistributedDataParallel) else model
65:     raw_model = getattr(raw_model, '_orig_mod', raw_model)
66:     state_dict = raw_model.state_dict()
67:     torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
68:     lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
69:     model.train()
```

逐行解释：

- `is_main_process()`：只保存一次（避免多 rank 写同一文件）。
- `model.eval()`：避免 dropout 等影响保存权重（对保存并无必要，但保持一致性）。
- `moe_suffix`：文件名区分 dense/moe。
- 保存两类文件：
  1. `args.save_dir/*.pth`：仅包含 half CPU 的 `state_dict`，作为权重快照供 `init_model(from_weight=...)` 加载
  2. `../checkpoints/*_resume.pth`（由 `lm_checkpoint` 创建）：包含 optimizer/scaler 等训练状态，供真正 resume

#### 2.2.8 一个常见边界：最后一段累积未满时的补步更新

```python
74: if last_step > start_step and last_step % args.accumulation_steps != 0:
75:     scaler.unscale_(optimizer)
76:     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
77:     scaler.step(optimizer)
78:     scaler.update()
79:     optimizer.zero_grad(set_to_none=True)
```

- 如果最后的 step 没刚好落在 `accumulation_steps` 的整除位置，必须手动执行一次 step，否则最后的梯度不会落地。

---

### 2.3 `__main__`：逐行解析完整预训练启动

#### 2.3.1 args 与核心超参与数据

```python
83: parser.add_argument("--epochs", ...)
84: ...
98: --max_seq_len=340
99: --use_moe (0/1)
100: --data_path=../dataset/pretrain_t2t_mini.jsonl
101: --from_weight=none
102: --from_resume=0
...
```

要点：

- `use_moe` 决定 `MiniMindConfig` 的结构（dense 或 MoE）。
- `dtype` 选择 AMP autocast 的精度策略。

#### 2.3.2 分布式与随机种子

```python
109: local_rank = init_distributed_mode()
110: if dist.is_initialized(): args.device = f"cuda:{local_rank}"
111: setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
```

- DDP 模式下，每个 rank seed 不同（但同一 rank 内可复现）。

#### 2.3.3 模型 config、checkpoint 是否 resume

```python
114: os.makedirs(args.save_dir, exist_ok=True)
115: lm_config = MiniMindConfig(..., use_moe=bool(args.use_moe))
116: ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

- 只有当 `from_resume==1` 才会读取 `*_resume.pth`。

#### 2.3.4 autocast/GradScaler 策略

```python
119: device_type = "cuda" if "cuda" in args.device else "cpu"
120: dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
121: autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
136: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
```

- CPU：不用 autocast。
- CUDA：
  - bfloat16 不一定需要 GradScaler（通常更稳定）
  - float16 才启用 GradScaler（避免 underflow）。

#### 2.3.5 init_model、dataset、optimizer

```python
133: model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
134: train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
135: train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
137: optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

- 非 DDP：`train_sampler=None`，后面用 `indices = torch.randperm(...)`。
- DDP：用 `DistributedSampler(train_ds)` 切分数据。

#### 2.3.6 从 resume 读取状态

```python
140: start_epoch, start_step = 0, 0
141: if ckp_data:
142:    model.load_state_dict(ckp_data['model'])
143:    optimizer.load_state_dict(ckp_data['optimizer'])
144:    scaler.load_state_dict(ckp_data['scaler'])
145:    start_epoch = ckp_data['epoch']
146:    start_step = ckp_data.get('step', 0)
```

关键：

- `lm_checkpoint` 可能已经对 `step` 做了 world_size 对齐缩放。

#### 2.3.7 编译与 DDP wrapper

```python
149: if args.use_compile == 1: model = torch.compile(model)
152: if dist.is_initialized():
153:    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
154:    model = DistributedDataParallel(model, device_ids=[local_rank])
```

- 忽略 buffer `freqs_cos/freqs_sin`：
  - 因为它们是非参数 buffer，某些场景可能引起 DDP 同步/广播开销或 mismatch。

#### 2.3.8 每 epoch 的 skip 续训

```python
157: for epoch in range(start_epoch, args.epochs):
158:    train_sampler and train_sampler.set_epoch(epoch)
159:    setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
160:    skip = start_step if (epoch == start_epoch and start_step > 0) else 0
161:    batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
162:    loader = DataLoader(train_ds, batch_sampler=batch_sampler, ...)
163:    if skip > 0: train_epoch(..., len(loader)+skip, start_step, wandb)
164:    else: train_epoch(..., len(loader), 0, wandb)
```

- `SkipBatchSampler` 的 skip 与 `start_step` 完整对应：
  - loader 的 enumerate 从 `start_step+1` 开始
  - iters 里也用 `len(loader)+skip` 来保持进度计算一致。

---

## 3. 全量 SFT：`train_full_sft.py`

SFT 脚本几乎与预训练一致，差异主要在：

- dataset 换成 `SFTDataset`（labels 构造不同）
- 保存的权重前缀 `save_weight` 默认 `full_sft`

本节我只覆盖“不同点 + 训练循环逐行关键差异”，因为训练循环其余部分完全同构。

### 3.1 dataset/init：`SFTDataset` 与 labels 语义差异

```python
16: from dataset.lm_dataset import SFTDataset
...
134: model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
135: train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

由于 `SFTDataset` 的 `labels` 在非 assistant token 位置为 `-100`，因此：

- `MiniMindForCausalLM.forward` 的 `cross_entropy(... ignore_index=-100)` 会自动忽略 prompt 部分监督。

从实现角度看：训练循环无需改动，只是 labels 值不同。

### 3.2 `train_epoch` 训练步：与预训练逐行一致

`train_epoch` 里关键逻辑同样是：

```python
res = model(input_ids, labels=labels)
loss = res.loss + res.aux_loss
loss = loss / args.accumulation_steps
scaler.scale(loss).backward()
if step % args.accumulation_steps == 0: unscale + clip + scaler.step + scaler.update + zero_grad
```

因此本节不重复所有细节解释，而强调：

- `res.loss` 对应 SFT 的 CE loss（只在 assistant token 上有效）
- `res.aux_loss` 仍按 MoE 层的 router 正则项贡献（dense 时通常为 0）

---

## 4. 本章小结（你需要“能复核”的点）

1. 预训练与全量 SFT 的训练循环骨架完全一致，差异主要来自 Dataset 的 `labels` 构造。
2. 梯度累积：
   - 用 `loss/accumulation_steps` 保持有效梯度尺度
   - 只有在 `step % accumulation_steps==0` 时才 `unscale/clip/step/update/zero_grad`
3. checkpoint：
   - `args.save_dir/*.pth`：半精度 CPU 权重快照
   - `../checkpoints/*_resume.pth`：包含 optimizer/scaler/world_size/epoch/step 等 resume 数据
4. DDP：
   - 由 `init_distributed_mode` + `RANK/LOCAL_RANK` 控制
   - 并忽略 `freqs_cos/freqs_sin` 这类 buffer 以避免 DDP 同步问题。

下一章我会生成 `06_LoRA 微调训练.md`，逐行解析 `train_lora.py`：如何 `apply_lora` 后冻结非 LoRA 参数、只训练 LoRA 子模块，并用 `save_lora` 保存“只包含 lora 的 state_dict”。

