# 12_AgentRL（ToolCall 多轮与奖励规则）

本章对应 `minimind_src/trainer/train_agent.py`，实现的是更复杂的 RL：agent 会在生成过程中多轮 ToolCall：

1. 模型先生成一段文本（其中可能包含 `<tool_call>...</tool_call>`）
2. 解析并执行工具，得到 tool response
3. 把 tool response 作为新上下文再让模型继续生成
4. 整个多轮交互结束后，根据“是否符合工具/格式/GT”等规则计算奖励
5. 最终把多轮交互打包成训练用的 `input_ids + completion_mask + old_per_token_logps + advantages`，然后用和 GRPO 类似的 ratio/KL + loss_type 分支进行反传

本章重点仍是你要求的“源码级逐行+张量维度追踪”，因此只对影响张量对齐的地方做深入维度解释：

- `rollout_single`：如何产出 `response_ids/response_mask/response_old_logps`
- `rollout_batch`：如何把一个 batch 的多个 samples 展开并打平
- `calculate_rewards`：奖励标量如何按 sample_idx 对齐到奖励向量
- `rl_train_epoch`：如何把多轮交互打包为固定长度张量并构造 `completion_mask/old_per_token_logps/per_token_logps` 对齐关系

---

## 1. 工具定义与 ToolCall 解析（用于模拟环境）

在 `train_agent.py` 前半段，定义了：

- `TOOLS`：函数签名列表（名称包括 `calculate_math`、`unit_converter`、`get_current_weather` 等）
- `MOCK_RESULTS`：这些函数在训练/测试时的模拟实现（不访问外部真实服务）
- `CHECK_ARGS`：对 tool arguments 做参数校验

关键解析函数：

```python
76: def parse_tool_calls(text):
77:     calls = []
78:     for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
79:         try: calls.append(json.loads(m.strip()))
80:         except: pass
81:     return calls

83: def execute_tool(name, args):
84:     fn = MOCK_RESULTS.get(name)
85:     if not fn: return None
86:     try:
87:         signal.signal(signal.SIGALRM, ...)
88:         signal.alarm(1)
89:         return fn(args)
90:     except:
91:         return None
92:     finally:
93:         try: signal.alarm(0)
94:         except: pass
```

这部分不产生张量，但它决定了多轮 rollout 中 tool response 的生成内容，因此间接影响奖励与后续训练张量里的“哪些 token 在 completion_mask 中为 1”。

---

## 2. 多轮 rollout：`rollout_single`（关键 token/mask/old_logps 产物）

`rollout_single` 负责一次样本的多轮 ToolCall 交互，最终返回一整条“打包序列”的组件。

函数签名：

```python
97: def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
```

### 2.1 初始化返回容器（逐行+语义）

```python
98: all_outputs = []
99: prompt_ids = None
100: response_ids = []
101: response_mask = []
102: response_old_logps = []
104: final_context = ""
105: unfinished = False
106: open_thinking = random.random() < thinking_ratio
```

- `prompt_ids`：第一次进入模型时的上下文 token（只记录一次）
- `response_ids`：多轮交互中模型产生的所有 token + tool observation 的 token
- `response_mask`：与 `response_ids` 同长度的 0/1 标记，**用于训练时决定 completion_mask 中哪些 token 参与 policy loss**
- `response_old_logps`：与 `response_ids` 同长度的 old logprob（policy 的旧策略 logprob），但 tool observation 部分会填 0.0 占位

### 2.2 每一轮：生成 -> 过滤 pad/eos -> 写入 response_ids/mask/old_logps

核心循环：

```python
106: for turn in range(max_turns):
107:     context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
108:     inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
109:     context_ids = inputs["input_ids"][0].tolist()
110:     if prompt_ids is None: prompt_ids = context_ids

112:     rollout_result = rollout_engine.rollout(
113:         prompt_ids=inputs["input_ids"],
114:         attention_mask=inputs["attention_mask"],
115:         num_generations=1,
116:         max_new_tokens=max_new_tokens,
117:         temperature=0.8,
118:     )

119:     new_ids = rollout_result.completion_ids[0].tolist()
120:     new_logps = rollout_result.per_token_logps[0].tolist()
121:     if len(new_ids) != len(new_logps): ...

122:     pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
123:     new_ids = [t for t, _ in pairs]
124:     new_logps = [lp for _, lp in pairs]

125:     new_text = rollout_result.completions[0]
126:     all_outputs.append(new_text)
127:     response_ids.extend(new_ids)
128:     response_mask.extend([1] * len(new_ids))
129:     response_old_logps.extend(new_logps)
```

维度/长度关系（注意这里是 list 而非 tensor）：

- `new_ids` 与 `new_logps` 初始长度相等（各生成 token 一一对应）
- 通过过滤 pad/eos 后：
  - `len(new_ids) == len(new_logps)` 仍应成立
- 写入时：
  - `response_ids` 增加 `len(new_ids)` 个 token
  - `response_mask` 增加 `len(new_ids)` 个 1
  - `response_old_logps` 增加对应 `len(new_ids)` 个旧 logprob 标量

### 2.3 解析 tool_call，执行工具，拼 messages，并把 tool response 的 token 写入 response_ids/mask=0/old_logps=0.0

生成文本 `new_text` 之后：

```python
131: calls = parse_tool_calls(new_text)
132: if not calls: break
134: unfinished = turn == max_turns - 1
135: messages.append({"role": "assistant", "content": new_text})

136: for call in calls:
137:     name, raw = call.get("name",""), call.get("arguments",{})
...
141: result = execute_tool(name, raw)
142: result_str = json.dumps(result, ensure_ascii=False)[:2048]
143: messages.append({"role": "tool", "content": result_str})
```

然后把“观察上下文 observe_context”编码成 token，并写入 response_ids：

```python
145: observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
146: observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
147: current_len = len(prompt_ids) + len(response_ids)
148: obs_delta = observe_ids[current_len:]
149: response_ids.extend(obs_delta)
150: response_mask.extend([0] * len(obs_delta))
151: response_old_logps.extend([0.0] * len(obs_delta))
152: final_context = observe_context
```

这里最关键的是“对齐语义”：

- `obs_delta` 是工具结果加入上下文后，observe_context 相对当前已打包内容新增的 token
- 这些 token 来自外部观察（tool response），不属于策略模型直接选择的 action token
- 因此：
  - `response_mask` 在这些 token 上填 0（不参与 policy loss）
  - `response_old_logps` 填 0.0（old logprob 用不到）

这保证了最终 `completion_mask` 的含义：只对模型自身生成的 token 做 policy update。

### 2.4 返回结构

```python
154: final_output = all_outputs[-1] if all_outputs else ""
155: prompt_ids = prompt_ids or []
156: return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished
```

---

## 3. 批量 rollout：`rollout_batch`（展开维度关系）

`rollout_batch` 接收：

- `messages_batch`: 长度 B 的 list（每个 batch 样本的 messages）
- `tools_batch`: 长度 B 的工具签名
- `num_gen`: 每个样本生成次数

它做的是 double loop，把每个样本生成多个 completion，并全部打平到一条列表：

```python
167: for messages, tools in zip(messages_batch, tools_batch):
168:     for _ in range(num_gen):
169:         completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(...)
171:         all_completions.append(completion)
173:         all_prompt_ids.append(prompt_ids)
174:         all_response_ids.append(response_ids)
175:         all_response_masks.append(response_mask)
176:         all_response_old_logps.append(response_old_logps)
177:         all_turn_outputs.append(turn_outputs)
178:         all_unfinished.append(unfinished)

179: return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished
```

因此 rollout 阶段生成后这些列表长度为：

- `len(completions) = B * num_gen`
- 并且每个索引 `idx` 对应某个样本 `sample_idx = idx // num_gen`（见 reward 计算逻辑）。

---

## 4. reward 规则：`calculate_rewards`（标量 reward 与样本对齐）

函数签名：

```python
187: def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
```

初始化：

```python
188: rewards = torch.zeros(len(completions), device=device)  # [B*num_gen]
189: for idx, response in enumerate(completions):
190:     reward, answer = 0.0, response
191:     sample_idx = idx // num_gen
192:     tools = tools_batch[sample_idx]
193:     turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
194:     unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
```

维度：

- `rewards` 是 tensor `[B*num_gen]`
- 每个 `idx` 对应一个 completion，因此奖励直接对齐到 PPO/GRPO/AgentRL 的打平维度

根据是否检测到 tool_calls（解析工具标签）选择两套规则：

1. 无 tool_calls：
   - 根据 response 长度、`</think>` 规则、是否有 reward_model 等加/减分
   - 最后扣除重复惩罚 `rep_penalty(answer)`
2. 有 tool_calls：
   - 计算工具调用数量是否和 GT 工具数量一致（`tool_gap`）
   - 对未完成（`unfinished`）扣分
   - 通过 `validate_gt_in_text` 检查 tool 的最终答案是否包含 GT 里的关键值，给 GT 分

最终：

```python
217: rewards[idx] = max(min(reward, 3.0), -3.0)
238: return rewards
```

因此输出 reward 的维度就是：

- `rewards: [B*num_gen]`，供后续 `grouped_rewards = rewards.view(-1,num_generations)` 做 group 相对标准化。

---

## 5. RL 训练：`rl_train_epoch`（多轮 token pack + completion_mask 对齐）

这一部分是本章最核心的“张量维度追踪”：

- 把 `prompt_ids_batch` 与 `response_ids_batch` 按 sample 打包成 `input_ids: [N, L]`
- 构造 `full_response_masks: [N, L]` 并切出 `completion_mask: [N, L-1]`
- 构造 `old_per_token_logps: [N, L-1]`，保证与 `per_token_logps: [N, L-1]` 完全对齐

其中 `N = B * num_gen`。

### 5.1 rollout_batch 的输出到 list（维度关系）

在 `with torch.no_grad()` 中：

```python
249: completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(..., args.num_generations, ...)
```

因此：

- `len(prompt_ids_batch) = N`
- `len(response_ids_batch) = N`
- `len(response_masks_batch) = N`
- `len(response_old_logps_batch) = N`

### 5.2 逐样本 pack：构造 `ids/mask/old_logps`

对每个打平样本：

```python
252: prompts = [tokenizer.apply_chat_template(...) for m,t in zip(messages_batch, tools_batch)]
253: packed_samples = []
254: for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
255:     ids = p + r
256:     mask = [0] * len(p) + m
257:     old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
```

这里需要你明确语义：

- `ids`：最终模型输入序列 tokens（prompt + response + tool observation tokens）
- `mask`：与 `ids` 同长度
  - prompt 部分 mask=0（不参与训练）
  - response 部分 mask 来自 rollout_single 的 `response_mask`（action token=1，tool observation=0）
- `old_logps`：与后续 `per_token_logps` 对齐的“旧 logprob 序列”
  - 因为 `per_token_logps` 是基于 `input_ids[:,1:]` 产生的（预测 token 对应位置 t+1）
  - prompt 的 action 对应位置不参与，所以用 `0.0` pad 出来

还会做截断以保证总长度上界：

```python
258: if len(ids) > args.max_total_len:
259:     ids = ids[-args.max_total_len:]
260:     mask = mask[-args.max_total_len:]
261:     old_logps = old_logps[-(len(ids) - 1):]
```

### 5.3 padding 形成张量：`input_ids/ full_response_masks/ old_per_token_logps`

```python
264: seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
265: max_len = seq_lens.max().item()
266: input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
268: full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
269: old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
```

维度总结（本次核心）：

- `input_ids: [N, max_len]`
- `full_response_masks: [N, max_len]`
- `old_per_token_logps: [N, max_len-1]`

后续会做：

- `logits = res.logits[:, :-1, :]` -> `[N, max_len-1, V]`
- `per_token_logps` 由 `input_ids[:,1:]` gather 得 `[N, max_len-1]`
- 因此 `old_per_token_logps` 的长度必须是 `max_len-1`，与之对齐。

---

## 6. 计算 per_token_logps（新策略）与 ref_per_token_logps（参考策略）

新策略：

```python
271: model_unwrapped = ...
272: with autocast_ctx:
273:     res = model_unwrapped(input_ids)
274:     aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0,...)
275:     logits = res.logits[:, :-1, :]  # [N, max_len-1, V]
276:     per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [N, max_len-1]
```

参考策略（仍用 compute_per_token_logps）：

```python
278: ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1)  # [N, max_len-1]
```

---

## 7. completion_mask：EOS 截断（逐行+维度）

```python
281: completion_mask = full_response_masks[:, 1:]  # [N, max_len-1]
282: is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()  # [N, max_len-1]
283: eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, dtype=torch.long)  # [N]
284: has_eos = is_eos.any(dim=1)  # [N]
285: eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
286: pos = torch.arange(completion_mask.size(1)).unsqueeze(0)  # [1, max_len-1]
287: completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()  # [N,max_len-1]
```

意义：

- completion_mask 原本来自 rollout_single：action token=1，observation token=0
- 再乘 EOS 截断因子：EOS 之后（含 EOS 位置之后）置 0，只保留 EOS 之前参与 loss。

token_counts 与 valid_rows：

```python
288: token_counts = completion_mask.sum(dim=1)  # [N]
289: valid_rows = token_counts > 0  # [N] bool
```

---

## 8. advantages（group 内相对化）与 ratio/kl（token 级）

advantages 计算：

```python
311: grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
312: mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [N]
313: std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [N]
314: advantages = (rewards - mean_r) / (std_r + 1e-4)  # [N]
```

token 级 ratio 与 KL：

```python
316: kl_div = ref_per_token_logps - per_token_logps  # [N, max_len-1]
317: per_token_kl = torch.exp(kl_div) - kl_div - 1  # [N, max_len-1]
318: ratio = torch.exp(per_token_logps - old_per_token_logps)  # [N, max_len-1]
```

注意：

- `advantages.unsqueeze(1)` 会把 `[N] -> [N,1]` 并广播到 `[N, max_len-1]`。

---

## 9. 两种 loss_type 分支：cispo vs grpo（逐 token）

cispo：

```python
319: if args.loss_type == "cispo":
320:     clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
321:     per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)  # [N,L-1]
```

grpo（默认）：

```python
323: else:
324:     clipped_ratio = torch.clamp(ratio, 1-args.epsilon, 1+args.epsilon)
325:     per_token_loss1 = ratio * advantages.unsqueeze(1)
326:     per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
327:     per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)  # [N,L-1]
```

最终 policy_loss 聚合（按 completion_mask 平均）：

```python
327: policy_loss = (
328:     ((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
329:     if valid_rows.any() else per_token_loss.sum()*0.0
)
```

维度：

- `per_token_loss*completion_mask`: `[N,L-1]`
- `.sum(dim=1)`: `[N]`
- `[...] [valid_rows]`: 只取有效行 -> `[n_valid]`
- 除以 `token_counts[valid_rows]`: `[n_valid]`
- `.mean()` -> scalar

总 loss：

```python
329: loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
330: loss.backward()
```

---

## 10. 训练步到 optimizer step（梯度累积）

```python
332: if step % args.accumulation_steps == 0:
333:     if args.grad_clip > 0: clip_grad_norm_(model.parameters(), args.grad_clip)
334:     optimizer.step(); scheduler.step(); optimizer.zero_grad()
335:     if is_main_process() and step % args.save_interval == 0: rollout_engine.update_policy(model)
```

以及最后补 step：

```python
365: if last_step > start_step and last_step % args.accumulation_steps != 0:
366:     clip_grad_norm_(...)
367:     optimizer.step(); scheduler.step(); optimizer.zero_grad()
368:     ...
```

---

## 11. 本章小结（你要能复核的维度与语义）

你应当能复核以下关键对齐链路：

1. rollout_single 语义：
   - `response_mask=1` 只标记 agent “自身生成的 action token”
   - tool observation token 在 rollout_single 中 mask=0
2. old logprob 与 token 对齐：
   - `old_per_token_logps: [N, max_len-1]`
   - 新策略 `per_token_logps: [N, max_len-1]`
   - ref 策略 `ref_per_token_logps: [N, max_len-1]`
3. completion_mask：
   - `full_response_masks: [N,max_len]` -> `completion_mask: [N,max_len-1]`
   - 再乘 EOS 截断，EOS 之后置 0
4. policy_loss 聚合：
   - token 级：`per_token_loss: [N,max_len-1]`
   - 序列级：`(sum over tokens / token_counts) -> [n_valid] -> mean -> scalar`

下一章（`13_推理与服务脚本...`）会把这些训练好的权重如何用 `generate` 对话、如何处理 streaming、如何输出 OpenAI 兼容格式都讲清楚（包括 ToolCall 的解析/回填与 SSE 流式返回）。

