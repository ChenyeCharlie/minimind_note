# 07_DPO（偏好对齐）

本章对应 `minimind_src/trainer/train_dpo.py`，实现的是 Direct Preference Optimization（DPO）。

你需要掌握两点：

1. `DPODataset` 输出的 chosen/rejected 序列如何在训练里拼成一个 batch，并在 DPO loss 内再切回 chosen/rejected。
2. `mask` 如何把“只监督 assistant 段”的 token 贡献聚合成每条序列的标量 logprob 总和，从而进入 DPO 的 log-ratio 计算。

---

## 0. 入口文件与核心函数列表

训练脚本定义了这些关键函数：

- `logits_to_log_probs(logits, labels)`
- `dpo_loss(ref_log_probs, policy_log_probs, mask, beta)`
- `train_epoch(...)`
- `__main__`：初始化分布式、加载策略模型/参考模型、DDP wrapper、resume、循环训练

---

## 1. Import（逐行解释）

```python
1: import os
2: import sys
...
7: import argparse
8: import time
9: import warnings
10: import torch
11: import torch.nn.functional as F
12: import torch.distributed as dist
13: from contextlib import nullcontext
14: from torch import optim
15: from torch.nn.parallel import DistributedDataParallel
16: from torch.utils.data import DataLoader, DistributedSampler
17: from model.model_minimind import MiniMindConfig
18: from dataset.lm_dataset import DPODataset
19: from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler
```

要点：

- 训练过程用 `F.log_softmax` / `F.logsigmoid`。
- `DPODataset` 提供 chosen/rejected 的张量与 mask。
- `init_model/lm_checkpoint/SkipBatchSampler` 这些来自 `trainer_utils.py`，用于工程一致性（见 `04` 章）。

---

## 2. `logits_to_log_probs`（逐行+张量维度）

```python
24: def logits_to_log_probs(logits, labels):
25:     # logits shape: (batch_size, seq_len, vocab_size)
26:     # labels shape: (batch_size, seq_len)
27:     # log_probs shape: (batch_size, seq_len)
28:     log_probs = F.log_softmax(logits, dim=2)
29:     log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
30:     return log_probs_per_token
```

维度追踪：

- 输入：
  - `logits: [N, S, V]`
  - `labels: [N, S]`
- 第 28 行：
  - `F.log_softmax(logits, dim=2)` -> `[N, S, V]`
- 第 29 行：
  - `labels.unsqueeze(2)`：`[N, S] -> [N, S, 1]`
  - `gather(dim=2, index=...)`：在 vocab 维取对应 token 的 logprob
  - 输出 `[N, S, 1]`，`squeeze(-1)` -> `[N, S]`

动机：

- 将每个位置“预测目标 token 的对数概率”抽取出来，供后续按 token mask 求和进入 DPO。

---

## 3. `dpo_loss`（逐行+chosen/rejected 切分机制）

```python
33: def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
34:     # ref_log_probs 和 policy_log_probs 都是 shape: (batch_size, seq_len)
35:     ref_log_probs = (ref_log_probs * mask).sum(dim=1)
36:     policy_log_probs = (policy_log_probs * mask).sum(dim=1)
37:
38:     # 将 chosen 和 rejected 数据分开
39:     batch_size = ref_log_probs.shape[0]
40:     chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
41:     reject_ref_log_probs = ref_log_probs[batch_size // 2:]
42:     chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
43:     reject_policy_log_probs = policy_log_probs[batch_size // 2:]
44:
45:     pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
46:     ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
47:     logits = pi_logratios - ref_logratios
48:     loss = -F.logsigmoid(beta * logits)
49:     return loss.mean()
```

### 3.1 token 级到序列级：mask 聚合

- `ref_log_probs: [N, S]`
- `policy_log_probs: [N, S]`
- `mask: [N, S]`（由 `DPODataset.generate_loss_mask` 产生，0/1）

第 35-36 行：

- `(ref_log_probs * mask)`：仍 `[N, S]`（mask 为 0 的位置贡献被清零）
- `.sum(dim=1)`：`[N, S] -> [N]`

因此得到：

- `ref_log_probs_seq: [N]`：每条样本在有效 token 上的 logprob 总和
- `policy_log_probs_seq: [N]`

### 3.2 chosen/rejected 切分

训练时会把：

- `x = cat([x_chosen, x_rejected], dim=0)`，所以在 batch 维上前半段对应 chosen，后半段对应 rejected

因此这里设：

- `batch_size = N = 2B`
- chosen：`[:N//2]` -> `[B]`
- rejected：`[N//2:]` -> `[B]`

### 3.3 DPO 核心 log-ratio

各项都是标量（每个样本的 batch 维上向量 `[B]`）：

- `pi_logratios = logp_policy(chosen) - logp_policy(rejected)` -> `[B]`
- `ref_logratios = logp_ref(chosen) - logp_ref(rejected)` -> `[B]`
- `logits = pi_logratios - ref_logratios` -> `[B]`
- `loss = -logsigmoid(beta * logits)` -> `[B]`
- `.mean()` -> 标量

动机：

- DPO 的直觉是：希望策略模型对 chosen 的相对偏好（相对 rejected 的 logprob 差）比参考模型更强。

---

## 4. `train_epoch`（逐行+张量维度：from Dataset 到 loss）

函数签名：

```python
52: def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
```

### 4.1 batch 解包与拼接

```python
56: for step, batch in enumerate(loader, start=start_step + 1):
58:     x_chosen = batch['x_chosen'].to(args.device)
59:     x_rejected = batch['x_rejected'].to(args.device)
60:     y_chosen = batch['y_chosen'].to(args.device)
61:     y_rejected = batch['y_rejected'].to(args.device)
62:     mask_chosen = batch['mask_chosen'].to(args.device)
63:     mask_rejected = batch['mask_rejected'].to(args.device)
64:     x = torch.cat([x_chosen, x_rejected], dim=0)
65:     y = torch.cat([y_chosen, y_rejected], dim=0)
66:     mask = torch.cat([mask_chosen, mask_rejected], dim=0)
```

维度追踪（假设 DataLoader batch 大小为 `B`）：

- 来自 `DPODataset` 的张量每个是 `[B, S]`，其中：
  - `S = max_seq_len - 1`（因为 dataset 里用 `chosen_input_ids[:-1]` / `[1:]`）
- 拼接后：
  - `x: [2B, S]`
  - `y: [2B, S]`
  - `mask: [2B, S]`

### 4.2 学习率更新（标量）

```python
68: lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
69: for param_group in optimizer.param_groups:
70:     param_group['lr'] = lr
```

### 4.3 forward：ref_model（no_grad）与 policy model

```python
72: with autocast_ctx:
73:     with torch.no_grad():
74:         ref_outputs = ref_model(x)
75:         ref_logits = ref_outputs.logits
76:     ref_log_probs = logits_to_log_probs(ref_logits, y)

78:     outputs = model(x)
79:     logits = outputs.logits
80:     policy_log_probs = logits_to_log_probs(logits, y)
```

维度追踪：

- `ref_outputs.logits: [2B, S, V]`
- `y: [2B, S]`
- `logits_to_log_probs` 输出：
  - `ref_log_probs: [2B, S]`
  - `policy_log_probs: [2B, S]`

注意：

- ref_model 是参考模型，冻结梯度以节省显存。
- `logits_to_log_probs` 不做 shift，而是依赖 dataset 的 `x/y` 构造保证对齐。

### 4.4 DPO loss + aux_loss + accumulation 缩放

```python
82: dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
83: loss = dpo_loss_val + outputs.aux_loss
84: loss = loss / args.accumulation_steps
```

维度：

- `dpo_loss_val` 标量
- `outputs.aux_loss` 标量（dense 通常为 0；MoE 为 router 正则项）
- 相加仍标量，然后除以 accumulation_steps 仍标量。

### 4.5 backward 与优化器 step

```python
86: scaler.scale(loss).backward()
88: if step % args.accumulation_steps == 0:
89:     scaler.unscale_(optimizer)
90:     torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
91:     scaler.step(optimizer)
92:     scaler.update()
93:     optimizer.zero_grad(set_to_none=True)
```

说明：

- 这里 clip 用 `model.parameters()`，DPO 训练通常是全量 dense/或 MoE，但不做 LoRA-only 的参数子集优化（和 PPO/GRPO 相比逻辑更简单）。

---

## 5. 保存 checkpoint（权重快照 + resume 状态）

```python
107: if (step % args.save_interval == 0 or step == iters) and is_main_process():
108:     model.eval()
109:     moe_suffix = '_moe' if lm_config.use_moe else ''
110:     ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
111:     raw_model = model.module if isinstance(model, DistributedDataParallel) else model
112:     raw_model = getattr(raw_model, '_orig_mod', raw_model)
113:     state_dict = raw_model.state_dict()
114:     torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
115:     lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
116:     model.train()
```

说明同 `04/05` 章：同样是权重快照 + resume pth，两套目的不同。

---

## 6. `__main__`：策略模型/参考模型加载、冻结与 DDP

### 6.1 seed、config、ckp resume

```python
157: local_rank = init_distributed_mode()
158: if dist.is_initialized(): args.device = f"cuda:{local_rank}"
160: setup_seed(...)

162: os.makedirs(args.save_dir, exist_ok=True)
164: lm_config = MiniMindConfig(..., use_moe=bool(args.use_moe))
165: ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
```

### 6.2 policy model 与 ref model

```python
181: model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
185: ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
186: ref_model.eval()
187: ref_model.requires_grad_(False)
```

这确保：

- policy model 参与 backward
- ref_model 只提供 logprob 计算的参考基线

### 6.3 optimizer（AdamW）与训练数据

```python
190: train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
192: scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
193: optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
```

### 6.4 DDP wrapper + buffer ignore

```python
205: if args.use_compile == 1: model = torch.compile(model)
208: if dist.is_initialized():
210:    model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
211:    model = DistributedDataParallel(model, device_ids=[local_rank])
```

注意：

- 这里仅对 `model` 包了 DDP，`ref_model` 没有包装；因为 ref_model `no_grad`，并不参与训练通信开销。

---

## 7. 本章小结（你应该能复核）

1. chosen/rejected 拼接规则：
   - `x = [chosen; rejected]` 在 batch 维拼接，然后在 `dpo_loss` 内按前半/后半切回。
2. token mask 聚合：
   - `ref_log_probs/policy_log_probs` 为 `[N,S]`
   - 乘 mask 并对 `dim=1` 求和得到序列标量 `[N]`，再用于 log-ratio。
3. logits_to_log_probs 对齐：
   - dataset 已构造 `x = input[:-1]`、`y = input[1:]`，所以在这里 logits 与 labels 不需要再 shift。

下一章（`08_知识蒸馏`）我会解析 `train_distillation.py`：它的关键不同是蒸馏使用 token 级 KL（温度缩放），而不是 DPO 的 log-ratio 偏好差。

