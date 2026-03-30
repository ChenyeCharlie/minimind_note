# 09_Rollout 引擎与 per-token logprob 计算

本章对应 `minimind_src/trainer/rollout_engine.py`。强化学习（PPO/GRPO/AgentRL）里最难统一的部分是：

1. 从 policy 模型生成 completion（并可多次生成）
2. 在生成的每个 token 位置上，计算“旧策略 old policy 下的 logprob”

本文件提供统一抽象 `RolloutEngine`，并实现两种后端：

- `TorchRolloutEngine`：本地调用 `model.generate` + 在本地再 forward 计算 per-token logprob
- `SGLangRolloutEngine`：通过 HTTP 调用 SGLang `/generate`，直接取回 `output_token_logprobs`

无论后端如何，最终都输出同一种张量结构给训练 loss：

- `output_ids`: `[B*num_gen, P+R]`（包含 prompt+completion）
- `completion_ids`: `[B*num_gen, R]`
- `per_token_logps`: `[B*num_gen, R]`（与 completion token 一一对应）

---

## 1. 工厂与抽象基类

### 1.1 `RolloutResult`（结果数据结构）

```python
37: @dataclass
38: class RolloutResult:
39:     output_ids: Tensor
40:     completion_ids: Tensor
41:     per_token_logps: Tensor
42:     completions: List[str]
```

维度约定：

- `output_ids` 与 `completion_ids` 用于训练里截取 attention mask/labels/masks（不同训练脚本略有切片）
- `per_token_logps` 是 RL loss 里的 old logprob/old_per_token_logps 基础输入

### 1.2 `RolloutEngine` 抽象类

```python
46: class RolloutEngine(ABC):
49:     @abstractmethod
50:     def rollout(...)-> RolloutResult: ...
53:     @abstractmethod
54:     def update_policy(self, model: torch.nn.Module): ...
```

- `rollout(...)`：负责生成与 logprob 计算
- `update_policy(...)`：
  - 对 torch 后端：把内部引用的 policy_model 更新为新模型
  - 对 sglang 后端：把权重保存到磁盘并通知 sglang 服务器加载

---

## 2. `compute_per_token_logps`（逐行+张量对齐）

这是全章最关键的张量对齐逻辑：给定 `input_ids`（包含 prompt+completion），计算“最后 `n_keep` 个 token”的 logprob。

```python
20: def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
21:     if n_keep <= 0:
22:         return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)
24:     unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
25:     input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
26:     logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
27:     per_token_logps = []
28:     for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
29:         ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
30:         per_token_logps.append(
31:             torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
32:         )
33:     return torch.stack(per_token_logps)
```

### 2.1 输入输出维度

假设：

- `input_ids: [B, T_total]`
- 希望保留最后 `n_keep=R` 个 completion token 的 per-token logprob

则返回：

- `per_token_logps: [B, R]`

### 2.2 第 26 行的对齐：为什么 `logits_to_keep=n_keep+1` 再 `[:, :-1, :]`

因果语言模型里：

- logits 在时间维 `t` 表示“预测 token `t` 的 logits 来自输入的前缀到 `t-1`”
- 如果你要对“最后 R 个 token（token index ... T_total-R ... T_total-1）”计算 logprob，
  你需要至少覆盖它们对应的 logits 位置

实现选择：

- `logits_to_keep = n_keep + 1`：让 forward 额外保留一个对齐所需的 logits 步
- `logits[:, :-1, :]`：丢掉最后一步 logits，让 logits 的时间维与 `ids_row` 的长度严格一致

因此 `logits` 在时间维的长度应为 `R`：

- 第 26 行后 `logits: [B, R, V]`

### 2.3 第 31 行的 gather：如何得到每个 token 的 logprob

循环内：

- `logits_row: [R, V]`
- `ids_row: [R]`（input_ids 的最后 R token id）

计算：

- `logits_row.log_softmax(dim=-1)`: `[R, V]`
- `ids_row.unsqueeze(1)`: `[R, 1]`
- `gather(dim=1, ...)`：从 vocab 维按 token id 取出对应 logprob -> `[R, 1]`
- `squeeze(1)`：得到 `[R]`

最终 stack：

- `torch.stack(per_token_logps)`：
  - 每个 batch 元素一个 `[R]`
  - -> `[B, R]`

动机：

- 训练 RL 时需要逐 token 的 old logprob，才能计算 PPO/GRPO ratio、KL、优势加权等。

---

## 3. `TorchRolloutEngine`（逐行+输出维度）

### 3.1 构造

```python
59: class TorchRolloutEngine(RolloutEngine):
60:     def __init__(self, policy_model, tokenizer, device="cuda", autocast_ctx=None):
61:         self.policy_model = policy_model
62:         self.tokenizer = tokenizer
63:         self.device = device
64:         self.autocast_ctx = autocast_ctx
```

### 3.2 `rollout`：generate -> completion_ids -> per_token_logps

```python
66: def rollout(...)-> RolloutResult:
67:     model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
69:     with torch.no_grad():
70:         output_ids = model.generate(
71:             input_ids=prompt_ids,
72:             attention_mask=attention_mask,
73:             max_new_tokens=max_new_tokens,
74:             do_sample=True,
75:             temperature=temperature,
76:             num_return_sequences=num_generations,
77:             pad_token_id=self.tokenizer.pad_token_id,
78:             eos_token_id=self.tokenizer.eos_token_id,
79:         )  # [B*num_gen, P+R]

81:     prompt_len = prompt_ids.size(1)
82:     completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]
```

维度追踪：

- prompt 输入：`prompt_ids: [B, P]`
- attention_mask：`[B, P]`
- 生成设置 `num_return_sequences=num_generations`：
  - 输出 batch 展开：`output_ids: [B*num_gen, P+R]`
- completion 切片：
  - `completion_ids: [B*num_gen, R]`

接着计算 per-token logprob：

```python
84:     from contextlib import nullcontext
85:     ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
86:     with ctx:
87:         per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1))
```

- `output_ids` 传入 compute_per_token_logps 作为 `input_ids`
- `n_keep=completion_ids.size(1)=R`
- 输出 `per_token_logps: [B*num_gen, R]`

最后文本解码：

```python
89: completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
90: return RolloutResult(output_ids, completion_ids, per_token_logps, completions)
```

---

## 4. `SGLangRolloutEngine`（HTTP 返回解析+padding 对齐）

Torch 后端在本地计算 logprob；SGLang 后端希望从服务端直接拿回 `output_token_logprobs`。

### 4.1 构造与 tokenizer

```python
98: def __init__(base_url, model_path, shared_ckpt_path="./sglang_ckpt", timeout=120):
102:     self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
103:     self.http = requests
```

### 4.2 `rollout`：去除左侧 padding、展开 num_gen、发请求

```python
105: def rollout(...):
106:     input_ids_list = []
107:     for ids, mask in zip(prompt_ids, attention_mask):
108:         valid_ids = ids[mask.bool()].tolist()
109:         input_ids_list.append(valid_ids)

111:     all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
```

维度语义：

- `prompt_ids: [B, P]`
- `attention_mask: [B, P]`
- `mask.bool()` 选出有效 token -> 每条样本一个长度不等的 `valid_ids: [P_valid_i]`
- 然后对每条 prompt 复制 `num_generations` 次：
  - 最终用于 HTTP 的 `input_ids` 数量为 `B*num_gen`

请求 payload：

```python
113: payload = {
114:     "input_ids": all_input_ids,
115:     "sampling_params": {
116:         "temperature": temperature,
117:         "max_new_tokens": max_new_tokens,
118:         "stop_token_ids": [eos_token_id] if eos_token_id else [],
119:     },
120:     "return_logprob": True,
121: }
```

### 4.3 解析服务端返回：output_ids 与 token logprobs

```python
126: results = resp.json()
130: all_output_ids, all_completion_ids, all_logprobs = [], [], []
132: prompt_len = prompt_ids.size(1)
134: for i, result in enumerate(results):
135:     meta = result.get("meta_info", {})
136:     completion_ids = meta.get("output_ids", result.get("output_ids", []))
137:     raw_logprobs = meta.get("output_token_logprobs", [])
...
151:     completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
```

这里关键是：

- 服务端直接给 `completion_ids`（completion 部分，不一定包含 prompt）
- `raw_logprobs` 解析成每个 token 的 logprob 列表 `logprobs: List[float]`

然后构造“全 output_ids”：

```python
146: prompt = all_input_ids[i]
147: full_output = prompt + completion_ids
148: all_output_ids.append(full_output)
149: all_completion_ids.append(completion_ids)
150: all_logprobs.append(logprobs)
```

### 4.4 padding：把不等长序列 pad 成统一张量

由于不同样本 `completion` 长度可能不同，所以需要 pad：

```python
153: device = prompt_ids.device
154: max_out_len = max(len(ids) for ids in all_output_ids)
155: max_comp_len = max(len(ids) for ids in all_completion_ids)
156: max_logp_len = max(len(lp) for lp in all_logprobs)

158: def pad_to_tensor(seqs, max_len, pad_val=0):
159:     return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)

161: return RolloutResult(
162:     output_ids=pad_to_tensor(all_output_ids, max_out_len),
163:     completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
164:     per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=0.0),
165:     completions=completions,
166: )
```

维度追踪：

- 输出：
  - `output_ids: [B*num_gen, max_out_len]`
  - `completion_ids: [B*num_gen, max_comp_len]`
  - `per_token_logps: [B*num_gen, max_logp_len]`

训练侧会按 `completion_ids.size(1)` 或 completion mask 再截断/对齐，所以这里按 max_len pad 是可行的。

---

## 5. `update_policy`（Torch vs SGLang）

### 5.1 Torch：仅引用更新

```python
92: def update_policy(self, model):
93:     self.policy_model = model
```

### 5.2 SGLang：保存权重到磁盘并通知加载

```python
168: def update_policy(self, model):
169:     unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
170:     abs_path = os.path.abspath(self.shared_ckpt_path)
171:     unwrapped.lm_head.weight = torch.nn.Parameter(unwrapped.lm_head.weight.clone())
172:     state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
173:     unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
174:     unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight
175:     self.tokenizer.save_pretrained(abs_path)
176:     resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
181:     return resp.status_code == 200
```

要点：

- `lm_head.weight` 与 `embed_tokens.weight` 在该模型里存在 weight tying（见 `01` 章），因此更新权重后需要把 tie 关系恢复：
  - `unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight`

---

## 6. `create_rollout_engine`（工厂）

```python
197: def create_rollout_engine(engine_type="torch", policy_model=None, tokenizer=None, device="cuda", autocast_ctx=None, ...):
207:     if engine_type == "torch":
208:         return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
209:     elif engine_type == "sglang":
210:         return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
```

---

## 7. 本章小结（你应掌握的可复核点）

1. `compute_per_token_logps(input_ids: [B,T], n_keep=R)` 返回 `[B,R]`
2. 对齐关键在：
   - `logits_to_keep=n_keep+1` 后再 `[:, :-1, :]`
   - 保证 logits 的时间维和 `input_ids[:, -n_keep:]` 的长度一致
3. Torch rollout 输出：
   - `output_ids: [B*num_gen, P+R]`
   - `completion_ids: [B*num_gen, R]`
   - `per_token_logps: [B*num_gen, R]`
4. SGLang rollout 输出在不同样本长度不等时通过 pad 成统一张量；训练侧会再配合 completion mask 做截断/有效 token 选择。

下一章 `10_PPO 强化学习` 会把本章输出的 `per_token_logps`、`completion_ids` 通过 ratio/KL/advantages 映射为 loss，并逐行追踪 mask 的维度变换。

