# 11_GRPO 强化学习（group 相对策略）

本章对应 `minimind_src/trainer/train_grpo.py`，实现了 Group Relative Policy Optimization（GRPO）。

与 PPO 的不同点在于：GRPO 不直接使用“外部奖励 -> advantage”那条轨迹，而是对同一个 prompt 下的 `num_generations` 个样本进行 **group 内相对化**：

- group 内计算均值/方差
- 每个样本的 advantage 用 `(r - mean) / (std + eps)` 标准化
- policy loss 仍然通过 token 级 ratio（新旧策略 logprob 差）与 completion mask 计算，并引入一个类似 KL 的正则项

---

## 1. 工具与奖励计算：`rep_penalty` + `calculate_rewards`

这两段逻辑与 `train_ppo.py` 基本一致；此处只强调输出维度：

- `calculate_rewards(prompts, responses, reward_model)`：
  - `prompts: list[str]` 长度 B
  - `responses: list[str]` 长度 B * `num_generations`
  - 输出 `rewards: Tensor`，形状为 `[B*num_gen]`

---

## 2. 核心训练函数：`grpo_train_epoch(...)`

函数签名：

```python
70: def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
```

本函数内部可以拆成 6 块：

1. prompt tokenization
2. rollout 生成 completion + 得到 old_per_token_logps
3. 计算 ref_per_token_logps
4. 计算 rewards（外部 reward model/规则）
5. group 内归一化得到 advantages + completion_mask（EOS 截断）
6. 构建 per-token ratio/kl 与两种 loss_type 分支计算 policy_loss，并反传更新

---

## 3. 从 Dataset 到 rollout 输入：prompt tokenization（逐行+维度）

训练 batch：

```python
71: for step, batch in enumerate(loader, start=start_step + 1):
72:     prompts = batch['prompt']  # list[str], length B
73:     prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
74:                               padding_side="left", add_special_tokens=False).to(args.device)
```

维度：

- `prompt_inputs["input_ids"]: [B, P]`
- `prompt_inputs["attention_mask"]: [B, P]`

可选截断：

```python
75: if args.max_seq_len:
76:     prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
77:     prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]
```

不改变张量阶数，只改变序列长度维度 P 的取值上限。

---

## 4. rollout：生成多个 completion（逐行+维度）

```python
79: rollout_result = rollout_engine.rollout(
80:     prompt_ids=prompt_inputs["input_ids"],
81:     attention_mask=prompt_inputs["attention_mask"],
82:     num_generations=args.num_generations,
83:     max_new_tokens=args.max_gen_len,
84:     temperature=0.8,
85: )
86: outputs = rollout_result.output_ids
87: completion_ids = rollout_result.completion_ids
88: completions = rollout_result.completions
89: old_per_token_logps = rollout_result.per_token_logps.to(args.device)
```

结合 `rollout_engine.py` 的约定，典型维度为：

- `outputs: output_ids`：
  - 形状 `[B*num_gen, P+R]`
  - 含 prompt 与 completion
- `completion_ids: [B*num_gen, R]`
- `old_per_token_logps: [B*num_gen, R]`

其中 `R` 是完成部分 token 数（由生成过程中 eos/pad 决定；rollout 引擎可能会 pad 到统一长度）。

---

## 5. actor 侧 per-token logprob：两种分支原因

```python
91: model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
92: with autocast_ctx:
93:     if use_sglang or lm_config.use_moe:
94:         res = model_unwrapped(outputs)
95:         aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
96:         logits = res.logits[:, :-1, :]
97:         per_token_logps = F.log_softmax(logits, dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1)[:, -completion_ids.size(1):]
98:     else:
99:         aux_loss = torch.tensor(0.0, device=args.device)
100:        per_token_logps = rollout_result.per_token_logps
```

### 5.1 分支 A：`use_sglang` 或 `use_moe` 时重算 per_token_logps

这里重算的关键是对齐输出与 labels 的 shift：

- `res.logits: [B*num_gen, P+R, V]`
- `logits[:, :-1, :]`：`[B*num_gen, P+R-1, V]`
- `outputs[:, 1:]`：`[B*num_gen, P+R-1]`（作为下一 token 的真实 id）
- gather 得到：
  - `gather(..., index=outputs[:,1:])` 后 shape `[B*num_gen, P+R-1]`
- 最后 `[:, -R:]` 截取 completion 部分：
  - 得到 `per_token_logps: [B*num_gen, R]`

为什么要重算？

- `use_sglang` 时 rollout engine 的 per-token logprobs 来自服务端，可能存在实现/截断差异。
- `use_moe` 时 rollout_result.per_token_logps 计算路径与 `aux_loss` 融合策略可能要求一致（本实现里直接用模型 forward 得到 res.aux_loss）。

### 5.2 分支 B：dense 且非 sglang 时直接复用 old 计算结果

- 直接用 rollout_engine 返回的 `rollout_result.per_token_logps`
- `aux_loss=0`（dense 情况）

---

## 6. ref_per_token_logps：始终用 compute_per_token_logps

```python
102: with torch.no_grad():
103:     ref_per_token_logps = compute_per_token_logps(ref_model, outputs, completion_ids.size(1))
```

维度：

- `outputs: [B*num_gen, P+R]`
- `n_keep = completion_ids.size(1)=R`
- 输出 `ref_per_token_logps: [B*num_gen, R]`

---

## 7. rewards 与 group 化：advantages（逐行+维度）

```python
104: rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

121: grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
122: mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
123: std_r = grouped_rewards.std(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
124: advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]
```

维度追踪：

- rewards `[B*num_gen]`
- view -> `[B, num_gen]`
- mean/std -> `[B]`
- repeat_interleave -> `[B*num_gen]`
- advantages 同形 `[B*num_gen]`

动机：

- 同一个 prompt 下，relative advantage 衡量“这个 completion 相对同组其它 completion 更好还是更差”。
- std 归一化让不同 prompt 的奖励尺度更可比。

---

## 8. completion_mask：EOS 截断（逐行+维度）

这一步用于 token 级 loss 里只统计有效 completion token。

```python
126: is_eos = completion_ids == tokenizer.eos_token_id  # [B*num_gen, R]
127: eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=args.device)
128: eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
129: completion_mask = (torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)).int()  # [B*num_gen, R]
```

维度：

- `is_eos: [N, R]`，N=B*num_gen
- `eos_idx: [N]`，默认取 R（表示“没有 eos 则全 token 有效”）
- `arange(R).expand(N,-1)`: `[N,R]`
- `eos_idx.unsqueeze(1)`: `[N,1]`
- <= 比较 -> `[N,R]`，转 int 得 completion_mask

动机：

- 如果某序列出现 EOS，则 `completion_mask` 在 EOS 位置及之前为 1；EOS 后为 0。
- 若无 EOS，则 `eos_idx=R`，比较条件 `pos<=R` 对 0..R-1 都为 True，使全 completion token 都参与 loss。

---

## 9. KL-like 正则与 ratio：token 级核心量（逐行+维度）

```python
131: kl_div = ref_per_token_logps - per_token_logps
132: per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]
133: ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]
```

维度：

- `per_token_logps, old_per_token_logps, ref_per_token_logps` 全为 `[N,R]`
- 所以 `kl_div`, `per_token_kl`, `ratio` 都是 `[N,R]`

解释：

- `ratio = exp(new_logp - old_logp)`：PPO/GRPO 里的策略比率（逐 token）
- `per_token_kl`：使用近似形式（`exp(x) - x - 1`）把 KL 正则转成可逐 token 计算的项。

---

## 10. 两种 loss_type 分支：`cispo` vs `grpo`

### 10.1 `cispo` 分支

```python
134: if args.loss_type == "cispo":
135:     clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
136:     per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
```

维度：

- `advantages: [N]`，`advantages.unsqueeze(1) -> [N,1]`，与 `[N,R]` 广播 -> `[N,R]`
- `clamped_ratio: [N,R]`
- `per_token_loss: [N,R]`

注意 `.detach()`：

- `clamped_ratio` 对应的 ratio 部分在该分支不让梯度通过（实现层面相当于对比率做截断并止梯度）。

### 10.2 非 cispo（默认 `grpo`）分支

```python
137: else:
138:     clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
139:     per_token_loss1 = ratio * advantages.unsqueeze(1)
140:     per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
141:     per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
```

维度：

- `per_token_loss1/per_token_loss2: [N,R]`
- `torch.min(... )`: `[N,R]`
- 减去 KL 正则再取负：`per_token_loss: [N,R]`

解释：

- 这里是 PPO 风格 clip 的 GRPO 变体：用 `min` 实现保守更新（避免 ratio 过大/过小造成过更新）。

---

## 11. policy_loss 的 mask 聚合（逐行+维度）

```python
142: policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
143: loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
144: loss.backward()
```

维度：

- `per_token_loss: [N,R]`
- `completion_mask: [N,R]`（int，但乘法会自动变成 float）
- `per_token_loss * completion_mask`: `[N,R]`
- `.sum(dim=1)`: `[N]`
- `completion_mask.sum(dim=1)`: `[N]`（每条 completion 有效 token 数）
- 除法 -> `[N]`
- `.mean()`: 标量 `policy_loss`
- `aux_loss`：标量
- 合成 `loss`：标量，再除 accumulation_steps

---

## 12. 梯度累积与 optimizer/scheduler step（逐行）

```python
146: if step % args.accumulation_steps == 0:
147:     if args.grad_clip > 0:
148:         torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
149:     optimizer.step()
150:     scheduler.step()
151:     optimizer.zero_grad()
152:     if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)
```

以及最后补 step：

```python
193: if step > start_step and step % args.accumulation_steps != 0:
194:     ... optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

---

## 13. 本章小结（你应当掌握的“复核点”）

1. rollout 维度核心：
   - `outputs: [B*num_gen, P+R]`
   - `completion_ids: [B*num_gen, R]`
   - `old_per_token_logps: [B*num_gen, R]`
2. group 化优势：
   - `rewards: [B*num_gen] -> view(B,num_gen)`
   - `advantages: [B*num_gen]`
3. completion_mask：
   - 基于 EOS 位置得到 `[B*num_gen,R]` 的 0/1 有效 token 掩码
4. 逐 token ratio 与 KL 正则：
   - `ratio: exp(new_logp - old_logp) -> [B*num_gen,R]`
   - `per_token_kl` 与 `per_token_loss` 全是 `[B*num_gen,R]`
5. loss_type 差异：
   - `cispo`：ratio clamp + detach + 线性形式
   - `grpo`：PPO 风格 ratio clamp + `min` 保守更新

下一章 `12_AgentRL（ToolCall 多轮与奖励规则）` 会把 GRPO/PPO 这种单轮 completion 的 RL 进一步升级为 agent 的多轮 ToolCall 交互打包与 reward 验证，并解释它如何构造 completion_mask 与 loss 的 token 对齐。

