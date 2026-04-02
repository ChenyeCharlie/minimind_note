# 10_PPO 强化学习

本章对应 `minimind_src/trainer/train_ppo.py`，实现了基于 PPO 的强化学习训练（包含 critic/value 网络与 reward model）。

整体数据流（和 `09_Rollout` 形成闭环）：

- 输入：`RLAIFDataset` 给出 `prompts: list[str]`
- `rollout_engine.rollout(...)` 用 actor 生成 `gen_out`（prompt+completion tokens）与 `responses_text`
- `calculate_rewards(prompts, responses_text, reward_model)` 得到外部标量奖励 `rewards: [B]`
- 在同一批 rollout token 上：
  - 用 critic 预测每个 token 的 value：`old_resp_values: [B, R]`
  - 用 actor/ref 计算每个 token 的 old logprob：`old_resp_logp: [B, R]` 与 `ref_resp_logp: [B, R]`
  - 将外部奖励注入到每条序列最后一个有效 token，得到 `token_rewards: [B, R]`
  - 通过 GAE 计算 `advantages: [B, R]` 与 `returns: [B, R]`
- 最后进入 PPO 更新：
  - policy loss（clipped ratio + KL-like penalty）
  - value loss（value clipping）
  - 通过 `approx_kl > early_stop_kl` 触发 early stop（但仍保持 backward 闭环）

---

## 1. 工具函数：重复惩罚 `rep_penalty`

```python
29: def rep_penalty(text, n=3, cap=0.5):
30:     toks = re.findall(r"\w+|[^\w\s]", text.lower())
31:     grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
32:     return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0
```

- 输入是字符串 `text`
- 用 n-gram（默认 3-gram）衡量重复程度：
  - `len(grams)-len(set(grams))` 越大，重复越多
- 输出是标量浮点，最终从 reward 中扣除。

---

## 2. Critic 网络：`class CriticModel(MiniMindForCausalLM)`

### 2.1 构造：替换 head 为 value_head

```python
36: class CriticModel(MiniMindForCausalLM):
37:     def __init__(self, params):
38:         super().__init__(params)
39:         self.value_head = nn.Linear(params.hidden_size, 1)
```

维度：

- critic 输出的是每个位置一个 value 标量
- `value_head` 把 hidden `[... , hidden_size] -> [..., 1]`

### 2.2 forward：返回每个 token 的值

```python
42:     def forward(self, input_ids=None, attention_mask=None, **kwargs):
43:         outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
44:         hidden_states = self.model.norm(outputs[0])
47:         values = self.value_head(hidden_states).squeeze(-1)
48:         return values
```

维度（以输入 `gen_out: [B, P+R]` 为例）：

- `outputs[0]` 来自 `MiniMindModel` 的 forward，hidden_states：
  - `[B, P+R, C]`
- `self.model.norm` 不改变维度：
  - `[B, P+R, C]`
- `value_head`：
  - `[B, P+R, C] -> [B, P+R, 1]`
- `.squeeze(-1)`：
  - `[B, P+R, 1] -> [B, P+R]`

因此 critic 输出给 PPO 的 `values_seq` 是 `[B, P+R]`。

---

## 3. 奖励计算：`calculate_rewards(prompts, responses, reward_model)`

逐行逻辑（关键维度是输出 `rewards: [B]`）：

```python
51: def calculate_rewards(prompts, responses, reward_model):
52:     rewards = torch.zeros(len(responses), device=args.device)
55:     with torch.no_grad():
56:         reward_model_scores = []
56:         for i, (prompt, response) in enumerate(zip(prompts, responses)):
57:             pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
58:             matches = re.findall(pattern, prompt, re.DOTALL)
59:             messages = [{"role": role, "content": content.strip()} for role, content in matches]
60:             answer = response
61:             rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
62:             if '</think>' in response:
63:                 thinking_content, answer_content = response.split('</think>', 1)
64:                 rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
65:                 rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
66:                 answer = answer_content.strip()
67:             rewards[i] -= rep_penalty(answer)
69:             score = reward_model.get_score(messages, answer)
70:             reward_model_scores.append(score)
72:         reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
73:         rewards += reward_model_scores
74:     return rewards
```

要点：

- 奖励是由三部分构成：
  1. 文本长度奖励/惩罚（`len(response)` 在阈值区间内加 0.5，否则减 0.5）
  2. 若存在 `</think>` 标签：
     - 对 thinking_content 的长度与闭合次数做奖励/惩罚
  3. 对 answer 字面重复 n-gram 惩罚：`rewards[i] -= rep_penalty(answer)`
  4. Reward Model 评分：`reward_model.get_score(messages, answer)` 再加进 rewards
- `reward_model.get_score(...)` 在 `trainer_utils.LMForRewardModel` 里把 messages 拼接成 evaluation prompt，然后调用 `model.get_score(...)` 并裁剪到 `[-3,3]`。

输出维度：

- `rewards: [B]`（B=prompts 的数量）

---

## 4. PPO 核心：`ppo_train_epoch(...)`（逐行+维度）

本函数是最大块。下面按 PPO 计算图分阶段解释，并在关键处写出张量维度变换。

### 4.1 初始化训练状态

```python
79: def ppo_train_epoch(...):
79:     actor_model.train()
80:     critic_model.train()
81:     grad_accum_step = 0
```

---

## 4.2 rollout -> response -> 基础张量构造

### 4.2.1 prompt tokenization

```python
83: for step, batch in enumerate(loader, start=start_step + 1):
84:     prompts = batch["prompt"]  # list[str], length B
85:     enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
86:                     max_length=args.max_seq_len, padding_side="left").to(args.device)
87:     prompt_length = enc.input_ids.shape[1]
```

维度：

- `enc.input_ids: [B, P]`
- `enc.attention_mask: [B, P]`
- `P = prompt_length`（padding 到统一长度）

### 4.2.2 rollout 生成 token

```python
89: rollout_result = rollout_engine.rollout(
90:     prompt_ids=enc.input_ids,
91:     attention_mask=enc.attention_mask,
92:     num_generations=1,
93:     max_new_tokens=args.max_gen_len,
94:     temperature=0.8,
95: )
96: gen_out = rollout_result.output_ids
97: responses_text = rollout_result.completions
98: rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]
```

维度（因为 `num_generations=1`）：

- `gen_out: [B, P+R]`
- `responses_text` 是长度为 B 的 list[str]
- `rewards: [B]`

---

## 4.3 生成序列内“参与训练的 response 区间 mask”

这一块决定了 PPO loss 只在 completion（response）token 上优化，而 prompt 部分不参与。

### 4.3.1 构造 labels（右移 1）

```python
114: full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R]
115: labels = gen_out[:, 1:].clone()  # [B, P+R-1]
116: seq_len, resp_start = gen_out.size(1) - 1, prompt_length - 1
117: resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= resp_start
118: final_mask = (resp_mask & (~labels.eq(tokenizer.pad_token_id))).float()  # [B, P+R-1]
```

维度推导：

- `gen_out` 长度为 `P+R`
- `labels = gen_out[:,1:]`：
  - 长度变成 `P+R-1`
- `seq_len = P+R-1`
- `resp_start = P-1`
- `resp_mask`：
  - `arange(seq_len)` 得到 `[seq_len]`，unsqueeze -> `[1, seq_len]`
  - 与 `>= resp_start` 得到 bool mask `[1, seq_len]`，广播到 `[B, seq_len]`
- `labels.eq(pad)` 在 padding 位置为 True，`~labels.eq(pad)` 得到 valid tokens
- `final_mask: [B, P+R-1]`（虽然在后续 rollout 阶段并未直接用它，但它体现了 response 区间定义）

### 4.3.2 response 子序列切片（得到 `[B,R]`）

```python
119: B = len(prompts)
120: resp_labels = labels[:, resp_start:]  # [B, R]
121: resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
122: resp_pad_mask = ~resp_labels.eq(tokenizer.pad_token_id)  # [B,R] bool
123: resp_lengths = resp_pad_mask.sum(dim=1); eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
124: has_eos = eos_mask.any(dim=1); eos_pos = torch.argmax(eos_mask.int(), dim=1)
125: resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
126: resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()  # [B,R]
127: resp_value_mask = resp_policy_mask.clone()
```

维度追踪：

- `resp_labels: [B, R]`
- `resp_idx`: `[1, R]`，广播到 `[B,R]`
- `resp_pad_mask: [B,R]`
- `resp_lengths: [B]`（每条 response 的有效长度；若遇到 EOS 则以 EOS 位置为截断点）
- `resp_lengths.unsqueeze(1): [B,1]`
- `resp_policy_mask: [B,R]`，0/1 float
- `resp_value_mask` 与 policy mask 相同

动机：

- PPO policy/value loss 只对每条序列的有效 response token（直到 EOS 或 padding 前）求平均。

---

## 4.4 Rollout 阶段（no_grad）：old logprob 与 old value

这一阶段只做推理，目的是得到 PPO 需要的 old 基准量：

- `old_resp_values: [B,R]`
- `old_resp_logp: [B,R]`
- `ref_resp_logp: [B,R]`

### 4.4.1 critic -> old_resp_values

```python
129: with torch.no_grad():
130:     critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
131:     values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
132:     old_resp_values = values_seq[:, resp_start:-1] * resp_value_mask
```

维度：

- `values_seq: [B, P+R]`
- `values_seq[:, resp_start:-1]`：
  - resp_start=P-1
  - 截到 -1 排除最后一个位置
  - 得到 `[B, R]`
- 乘 `resp_value_mask: [B,R]` -> `[B,R]`

### 4.4.2 actor -> old_resp_logp（逐 token logprob）

```python
134: actor_for_rollout = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
135: with autocast_ctx:
136:     logits = actor_for_rollout(input_ids=gen_out, attention_mask=full_mask).logits
138: old_resp_logp = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)[:, resp_start:]
```

维度追踪：

- `logits: [B, P+R, V]`
- `logits[:, :-1] -> [B, P+R-1, V]`
- `labels.unsqueeze(-1): [B, P+R-1, 1]`
- gather dim=2（vocab） -> `[B, P+R-1, 1]`
- squeeze -> `[B, P+R-1]`
- `[:, resp_start:]` -> `[B,R]`

### 4.4.3 ref -> ref_resp_logp

```python
140: ref_logp_all = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1)\
141:              .gather(2, labels.unsqueeze(-1)).squeeze(-1)
141: ref_resp_logp = ref_logp_all[:, resp_start:]
```

- 同 old_resp_logp：`ref_resp_logp: [B,R]`

---

## 4.5 把外部奖励注入到每条 response 的最后 token

```python
142: token_rewards = torch.zeros_like(old_resp_logp)  # [B,R]
143: last_idx = resp_lengths - 1  # [B]
144: token_rewards[torch.arange(B, device=args.device), last_idx] += rewards  # [B]
```

维度：

- `old_resp_logp: [B,R]` -> `token_rewards: [B,R]`
- `last_idx` 是每条序列最后一个有效 token 的位置索引
- 高级索引把 `rewards[b]` 加到 `token_rewards[b,last_idx[b]]`
- 其它位置保持 0

动机：

- 本实现把整个序列奖励作为“末尾奖励”，通过 GAE 把末尾奖励回传到所有前缀 token 的优势估计。

---

## 4.6 GAE：advantages 与 returns（逐行+维度）

```python
146: gen_len = old_resp_values.size(1)  # R
147: lastgaelam = torch.zeros(B, device=args.device)
148: advs_rev = []
149: for t in reversed(range(gen_len)):
150:     nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
151:     delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
152:     lastgaelam = delta + args.gamma * args.lam * lastgaelam
153:     advs_rev.append(lastgaelam)
154: advantages = torch.stack(advs_rev[::-1], dim=1)  # [B,R]
155: returns = advantages + old_resp_values  # [B,R]
```

维度推导：

- 循环 t 对每个 token 计算标量 advantage：
  - `token_rewards[:, t]`: `[B]`
  - `nv`: `[B]` 或 0.0 标量（t=末尾时）
  - `old_resp_values[:, t]`: `[B]`
  - delta / lastgaelam 都是 `[B]`
- 最终 `torch.stack` 把 R 个 `[B]` 堆叠 -> `[R,B]` 再按 dim=1 得到 `[B,R]`

动机：

- GAE 结合了 TD error 的 delta，并用 `lam` 做 bias-variance trade-off。

### 4.6.1 Advantage 标准化（mask 加权）

```python
155: adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
156: adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
157: advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask
```

输出仍是 `[B,R]`，mask 的位置被置 0，保证无效 token 不影响 loss。

---

## 4.7 PPO 更新：clipped policy + value clipping

### 4.7.1 minibatch 划分

```python
159: mb_size = max(1, min(args.mini_batch_size, B))
170: for ppo_epoch in range(args.ppo_update_iters):
173:     b_inds = torch.randperm(B, device=args.device)
174:     for i in range(0, B, mb_size):
175:         inds = b_inds[i:i + mb_size]
```

- `inds` 使得当前更新 batch 大小为 `MB = len(inds)`。

### 4.7.2 critic forward -> mb_resp_values

```python
177: mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
178: mb_resp_values = mb_values_seq[:, resp_start:-1]
```

- `mb_values_seq`: `[MB, P+R]`
- `mb_resp_values`: `[MB, R]`

### 4.7.3 actor forward -> mb_resp_logp + aux_loss

```python
180: with autocast_ctx:
181:     res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
182:     aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

184: mb_logp_all = F.log_softmax(res.logits[:, :-1], dim=-1)\
185:     .gather(2, labels[inds].unsqueeze(-1)).squeeze(-1)
186: mb_resp_logp = mb_logp_all[:, resp_start:]
```

- `res.logits`: `[MB, P+R, V]`
- `mb_resp_logp`: `[MB,R]`
- `aux_loss`: 标量

### 4.7.4 PPO policy ratio 与 early stop（approx_kl）

```python
187: log_ratio = mb_resp_logp - old_resp_logp[inds]  # [MB,R]
188: approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

191: approx_kl_val = approx_kl.detach().clone()
192: if dist.is_initialized(): dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
195: if approx_kl_val > args.early_stop_kl: stop_ppo = True

198: ratio = torch.exp(log_ratio)  # [MB,R]
199: clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1))
```

维度：

- `approx_kl`: scalar
- `ratio`: `[MB,R]`
- `clipfrac`: scalar（mask 加权统计）

### 4.7.5 KL-like penalty：`kl_ref_penalty`

```python
201: kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
202:                      * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
```

令 `x = ref_resp_logp - mb_resp_logp`：

- `x`: `[MB,R]`
- 表达式 `exp(x) - x - 1` 是 `KL` 泰勒近似形状的一种实现形式
- 与 `resp_policy_mask` 相乘后求均值，得到标量惩罚

### 4.7.6 policy_loss（clipped surrogate + kl_penalty）

```python
203: policy_loss = (
204:     torch.max(
205:         -advantages[inds] * ratio,
206:         -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
207:     ) * resp_policy_mask[inds]
208: ).sum() / resp_policy_mask[inds].sum().clamp(min=1) + args.kl_coef * kl_ref_penalty
```

维度：

- `advantages[inds]`: `[MB,R]`
- `ratio/clamp(ratio)`: `[MB,R]`
- 两项取 `torch.max`：仍 `[MB,R]`
- 乘 mask 并 sum/mean：标量

符号说明：

- 这里是 `-advantages * ratio` 的形式，等价于 PPO 中使用 surrogate objective 的一种写法；最终通过最小化 `policy_loss` 达到最大化策略优势的效果。

### 4.7.7 value_loss（value function clipping）

```python
207: value_loss = 0.5 * (torch.max(
208:     (mb_resp_values - returns[inds]) ** 2,
209:     (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
210:                   old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2
211: ) * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)
```

维度：

- `mb_resp_values, returns, old_resp_values`: `[MB,R]`
- clamp 后仍 `[MB,R]`
- squared -> `[MB,R]`
- 乘 mask -> `[MB,R]`，sum/mean -> scalar

---

## 4.8 stop_ppo 的特殊处理：保持 DDP backward 闭环

```python
215: if stop_ppo:
216:     loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
217: else:
218:     loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
219: loss.backward()
```

- 早停时把 loss 乘 0，但仍调用 `backward()`，保证每个 DDP rank 都走同样的通信/梯度同步路径，避免死锁。

---

## 4.9 梯度累积与优化器 step

```python
233: grad_accum_step += 1
233: if grad_accum_step % args.accumulation_steps == 0:
234:     clip_grad_norm_(actor_model.parameters(), args.grad_clip)
235:     clip_grad_norm_(critic_model.parameters(), args.grad_clip)
236:     actor_optimizer.step(); critic_optimizer.step()
238:     actor_scheduler.step(); critic_scheduler.step()
240:     actor_optimizer.zero_grad()
241:     critic_optimizer.zero_grad()
```

- 先累积 backward，再在整除点 step。

训练末尾还有一个“最后不整除”的补 step：

```python
243: if grad_accum_step % args.accumulation_steps != 0:
244:   ... actor/critic step ...
```

---

## 4.10 rollout_engine.update_policy 与保存

在 PPO 更新后，主进程会更新 rollout 引擎所引用的 actor 权重：

```python
253: if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(actor_model)
```

保存模型（actor 与 critic + scheduler）：

```python
281: if (step % args.save_interval == 0 or step == iters) and is_main_process():
282:     actor_model.eval()
284:     ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
287:     torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
290:     lm_checkpoint(..., model=actor_model, optimizer=actor_optimizer,
293:                    scheduler=actor_scheduler, critic_model=critic_model,
294:                    critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
```

这里的意义：

- `.pth` 文件：actor 权重快照（供下次 init_model 加载）
- `lm_checkpoint`：保存 critic/optimizer/scheduler/scaler 等训练 resume 状态

---

## 5. __main__：PPO 启动与模块装配

### 5.1 构建 actor/ref/critic 与 reward model

关键片段：

```python
372: actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
376: ref_model, _ = init_model(lm_config, base_weight, device=args.device)
377: ref_model = ref_model.eval().requires_grad_(False)
381: critic_model = CriticModel(lm_config)
382: critic_model.load_state_dict(state_dict, strict=False)

384: reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
```

张量语义：

- actor/ref：都是 `MiniMindForCausalLM`（输出 logits）
- critic：输出 value logits（标量 value 而非 vocab logits）
- reward_model：外部评分接口，输出标量 reward

### 5.2 rollout 引擎

```python
386: rollout_engine = create_rollout_engine(engine_type=args.rollout_engine, policy_model=actor_model, tokenizer=tokenizer, ...)
```

如果 engine_type 是 `torch`，则 rollout 在本进程里 `generate` 并 forward 计算 old logprob；
如果是 `sglang`，则通过 HTTP 直接取回 per-token logprob（并 pad 到统一长度）。

### 5.3 dataset 与 data loader

```python
396: train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
400: loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
401: iters = len(loader_for_count)
```

`RLAIFDataset.__getitem__` 返回：

- `{'prompt': prompt, 'answer': ""}`

因此 DataLoader 默认 collate 会产生：

- `batch["prompt"]: list[str]`，满足 `prompts = batch["prompt"]`

---

## 6. 本章小结

1. response 区间切片定义：
   - `resp_start = prompt_length - 1`
   - `labels: [B,P+R-1]`，`resp_labels: labels[:, resp_start:] -> [B,R]`
2. old logprob 与 old value 的形状必须一致：
   - `old_resp_logp: [B,R]`
   - `old_resp_values: [B,R]`
3. advantages/returns：
   - token_rewards 只在 `last_idx=resp_lengths-1` 位置注入外部 `rewards: [B]`
   - 通过反向遍历得到 `advantages: [B,R]`、`returns: [B,R]`
4. PPO loss 全程通过 `resp_policy_mask/resp_value_mask` 做 token 级屏蔽与均值聚合。
5. `stop_ppo` 早停只是把 loss 乘 0，但仍 `backward()`，保证 DDP 不死锁。

下一章 `11_GRPO 強化学习（group 相对策略）` 会与 PPO 类似，但差别在：
- rollout 的 reward 以 group 方式聚合（均值/方差标准化）
- loss 采用 GRPO/cispo 风格的 ratio/clamp 形式

