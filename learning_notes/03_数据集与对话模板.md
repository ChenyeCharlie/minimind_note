# 03_数据集与对话模板（逐行+labels/mask 维度）

本章对应 `minimind_src/dataset/lm_dataset.py`，核心目标是把训练“要预测哪些 token”这件事彻底讲清楚：

- `input_ids` 的构造（prompt 字符串 -> tokenizer -> token id）
- `labels` / `loss_mask` 的构造（哪些位置参与 loss，哪些位置被忽略）
- 在不同训练范式（Pretrain/SFT/DPO/RL）中，这些 mask 的语义如何变化

注意：`Dataset.__getitem__` 返回的是**单样本**张量，维度一般是 `[T]`（而 DataLoader 会在 batch 维度上变成 `[B,T]`）。本章会在关键处显式指出这两层维度语义。

---

## 1. Import 与环境设置（逐行）

```python
1: from torch.utils.data import Dataset
2: import torch
3: import json
4: import os
5: import random
6: from datasets import load_dataset, Features, Sequence, Value
7: os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

- 第 1 行：PyTorch Dataset 基类。
- 第 6 行：HuggingFace `datasets` 用于从 json/jsonl 读取数据并可声明特征 schema（尤其 SFT 的结构）。
- 第 7 行：关闭 tokenizer 并行，避免在某些环境下出现多进程/多线程警告或不确定行为。

---

## 2. 对话预处理：`pre_processing_chat`

```python
9: def pre_processing_chat(conversations, add_system_ratio=0.2):
10:     # tool use 数据完整保留不做处理
11:     if any(conv.get('tools') for conv in conversations): return conversations
12:
13:     SYSTEM_PROMPTS = [
14:         "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
15:         "你是minimind，一个小巧但有用的语言模型。",
16:         "你是一个专业的AI助手，请提供有价值的回答。",
17:         "你是minimind，请尽力帮助用户解决问题。",
18:         "你是一个可靠的AI，请给出准确的回答。",
19:         "You are a helpful AI assistant.",
20:         "You are minimind, a lightweight intelligent assistant.",
21:         "You are a friendly chatbot. Please answer the user's questions carefully.",
22:         "You are a knowledgeable AI. Try your best to provide accurate information.",
23:         "You are minimind, a small but useful language model."
24:     ]
25:     # 概率性添加system
26:     if conversations[0].get('role') != 'system':
27:         if random.random() < add_system_ratio:
28:             return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
29:     return conversations
```

逐行语义：

- 第 10 行：如果对话中任意轮出现 `tools` 字段，则直接返回，不做系统提示插入。
  - 这是为了避免工具调用相关的结构被改写（否则 `tool_call`/`tools` 语义可能破坏）。
- 第 26-28 行：当第一个 message 的 `role` 不是 `system` 时，以 `add_system_ratio` 概率在最前面插入一个随机 system prompt。
  - 返回的是一个新列表：`[system_msg] + conversations`

维度与张量：

- 该函数只处理 Python dict/list，不涉及 tensor 维度。

---

## 3. 对话后处理：`post_processing_chat`

```python
31: def post_processing_chat(prompt_content, empty_think_ratio=0.2):
32:     # 以80%概率移除空思考标签
33:     if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
34:         prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
35:     return prompt_content
```

逐行语义：

- 第 33 行的概率：`random.random() > empty_think_ratio` 的概率是 `1 - empty_think_ratio`
  - 默认 `empty_think_ratio=0.2`，因此默认移除概率是 0.8（与注释一致）
- 只删除完全空的思考标签形态：`<think>\n\n</think>\n\n`

---

## 4. 预训练数据集：`PretrainDataset`

### 4.1 init 与基本样本读取（逐行）

```python
37: class PretrainDataset(Dataset):
38:     def __init__(self, data_path, tokenizer, max_length=512):
39:         super().__init__()
40:         self.tokenizer = tokenizer
41:         self.max_length = max_length
42:         self.samples = load_dataset('json', data_files=data_path, split='train')
```

- `self.samples`：HuggingFace dataset split，类型为 json 条目集合。

### 4.2 __getitem__：CE labels 的对齐方式（逐行+维度）

```python
47:     def __getitem__(self, index):
48:         sample = self.samples[index]
49:         tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
50:         tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
51:         input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
52:         input_ids = torch.tensor(input_ids, dtype=torch.long)
53:         labels = input_ids.clone()
54:         labels[input_ids == self.tokenizer.pad_token_id] = -100
55:         return input_ids, labels
```

维度追踪（单样本）：

- tokenizer 输出 `tokens: List[int]` 长度最终控制为 `max_length-2`（因为要留 `bos/eos` 两个 token）
- `input_ids: [T]` 其中 `T=self.max_length`
- `labels: [T]`

语义：

- 预训练采用最标准的 causal LM：`labels` 与 `input_ids` 同 shape。
- pad 位置的 label 设置为 `-100`，这样模型 forward 内 `F.cross_entropy(..., ignore_index=-100)` 会跳过这些位置。

---

## 5. 全量 SFT 数据集：`SFTDataset`

### 5.1 init：特征 schema 与 BOS/EOS 序列（逐行+动机）

```python
58: class SFTDataset(Dataset):
59:     def __init__(self, jsonl_path, tokenizer, max_length=1024):
60:         super().__init__()
61:         self.tokenizer = tokenizer
62:         self.max_length = max_length
63:         features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
64:         self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
65:         self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
66:         self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
```

schema 动机：

- 对话数据里每轮 message 可能带 `reasoning_content/tools/tool_calls` 等字段。
- 通过 `Features` 显式声明，`datasets` 会更稳定地解析 json。

BOS/EOS 的特殊性（用于定位 assistant 段）：

- `self.bos_id`：不是单个 token id，而是一段 token 序列（BOS + `assistant\n`）。
  - 因为在 chat_template 下，一个 assistant 段开头并不一定是单 token，可能是多个 token 的拼接。
- `self.eos_id`：同理，用于定位 assistant 段的结束（`eos_token + '\n'`）。

### 5.2 create_chat_prompt：把 conversations 变成 prompt 字符串（逐行+动机）

```python
71:     def create_chat_prompt(self, conversations):
72:         messages = []
73:         tools = None
74:         for message in conversations:
75:             message = dict(message)
76:             if message.get("role") == "system" and message.get("tools"):
77:                 tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
78:             if message.get("tool_calls") and isinstance(message["tool_calls"], str):
79:                 message["tool_calls"] = json.loads(message["tool_calls"])
80:             messages.append(message)
81:         return self.tokenizer.apply_chat_template(
82:             messages,
83:             tokenize=False,
84:             add_generation_prompt=False,
85:             tools=tools
86:         )
```

逐行解释：

- 第 74-80 行：遍历原始 conversations：
  - 复制 dict，避免原数据被就地修改
  - 如果遇到 system 且存在 `tools`：
    - 若 `tools` 是 json 字符串，就 `json.loads` 解析为 Python 对象
    - 否则直接使用原对象
  - 如果遇到 `tool_calls` 且它是 str，也做 json 解析
- 第 81-86 行：
  - 交给 tokenizer 的 `apply_chat_template(...)` 生成最终 prompt 文本
  - `tokenize=False`：只生成字符串，不立即转 token
  - `add_generation_prompt=False`：SFT 训练本身已经包含 assistant 的答案片段，因此不额外追加“下一步要生成”的标记
  - `tools=tools`：把工具签名喂给模板（模板里会决定 tool_call 的文本结构）

### 5.3 generate_labels：把 prompt 的 assistant 段落映射为 labels（逐行+维度）

```python
88:     def generate_labels(self, input_ids):
89:         labels = [-100] * len(input_ids)
90:         i = 0
91:         while i < len(input_ids):
92:             if input_ids[i:i + len(self.bos_id)] == self.bos_id:
93:                 start = i + len(self.bos_id)
94:                 end = start
95:                 while end < len(input_ids):
96:                     if input_ids[end:end + len(self.eos_id)] == self.eos_id:
97:                         break
98:                     end += 1
99:                 for j in range(start, min(end + len(self.eos_id), self.max_length)):
100:                     labels[j] = input_ids[j]
101:                 i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
102:             else:
103:                 i += 1
104:         return labels
```

这是 SFT 的关键：它决定“哪些 token 算训练目标”。

维度语义：

- `input_ids: List[int]`，长度为 `T=max_length`
- 返回 `labels: List[int]`，同样长度 `[T]`

具体策略：

- 初始时所有位置 `labels=-100`（完全不参与 loss）
- 扫描输入 token 序列，寻找 assistant 段起始子串 `self.bos_id`
  - 一旦命中：
    - `start = i + len(bos_id)`：assistant 段内容开始位置（不把开头 bos_id 本身算进 loss）
    - 向后移动 `end`，直到找到 EOS 子串 `self.eos_id`
    - 然后把 `[start, end + len(eos_id))` 范围内的 token 全部设为 `labels=input_ids`，等价于这段 token 都是要预测的目标
- `i` 跳到段落结束后，继续寻找下一段（多轮对话会出现多个 assistant 段）

动机：

- 对话式 SFT 通常不希望模型在“user/system 提示部分”做 next-token 监督（而是只监督 assistant 的输出片段），因此用 BOS/EOS 定位 assistant 输出区间来构造 labels。

### 5.4 __getitem__：最终返回 `[input_ids, labels]`（逐行+维度）

```python
106:     def __getitem__(self, index):
107:         sample = self.samples[index]
108:         conversations = pre_processing_chat(sample['conversations'])
109:         prompt = self.create_chat_prompt(conversations)
110:         prompt = post_processing_chat(prompt)
111:         input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
112:         input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
113:         labels = self.generate_labels(input_ids)
114:         # 调试打印（被注释）
119:         return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
```

维度追踪（单样本）：

- `input_ids: [T]`（T=max_length）
- `labels: [T]`
- DataLoader batch 后会变为：
  - `input_ids: [B, T]`
  - `labels: [B, T]`

---

---

## 6. 偏好对齐数据：`DPODataset`（逐行+loss_mask/切片维度）

`DPODataset` 处理 DPO 的数据格式：同一个问题给出 `chosen`（偏好）与 `rejected`（反偏好）两条对话，其中你只需要让模型在“偏好回答段”上获得更高概率。

### 6.1 init：存储 tokenizer 与 BOS/EOS 子串（逐行）

```python
122: class DPODataset(Dataset):
123:     def __init__(self, file_path, tokenizer, max_length=4096):
124:         super().__init__()
125:         self.tokenizer = tokenizer
126:         self.max_length = max_length
127:         self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
128:         self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
129:         self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
130:         self.samples = load_dataset('json', data_files=file_path, split='train')
```

- `bos_id`/`eos_id` 都是“序列 token id 列表”，用于在编码后的 `input_ids` 中定位 assistant 段范围。
- `self.samples` 每条样本包含 `chosen`、`rejected` 两个对话（在 `__getitem__` 中读取）。

### 6.2 generate_loss_mask：assistant 段内置 1（逐行+维度语义）

```python
176: def generate_loss_mask(self, input_ids):
177:     loss_mask = [0] * len(input_ids)
178:     i = 0
179:     while i < len(input_ids):
180:         if input_ids[i:i + len(self.bos_id)] == self.bos_id:
181:             start = i + len(self.bos_id)
182:             end = start
183:             while end < len(input_ids):
184:                 if input_ids[end:end + len(self.eos_id)] == self.eos_id:
185:                     break
186:                 end += 1
187:             for j in range(start, min(end + len(self.eos_id), self.max_length)):
188:                 loss_mask[j] = 1
189:             i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
190:         else:
191:             i += 1
192:     return loss_mask
```

mask 的语义：

- 输入 `input_ids` 是长度为 `T=max_length` 的列表（注意：这里是 DPO 的每条 chosen/rejected 编码后序列）。
- `loss_mask: [T]`。
- 扫描到 assistant 段开头 `bos_id` 之后：
  - 从 `start=i+len(bos_id)` 开始，到首次遇到 `eos_id` 为止（包含 eos 子串长度）：
    - `loss_mask[j]=1`，表示这些 token 位置参与 loss。
- 对其它非 assistant 提示部分，mask 保持 0，不参与 DPO 的 chosen/rejected logprob 比较。

### 6.3 __getitem__：输出 x/y/mask（逐行+张量维度）

```python
135: def __getitem__(self, index):
136:     sample = self.samples[index]
137:     chosen = sample['chosen']
138:     rejected = sample['rejected']
139:     chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
142:     rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
148:     chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
151:     rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')
155:     chosen_input_ids = chosen_encoding['input_ids']
156:     chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)
158:     rejected_input_ids = rejected_encoding['input_ids']
159:     rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
160:     x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
161:     y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
162:     mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
163:     x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
164:     y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
165:     mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)
167:     return {...}
```

关键维度对齐（DPO 用的是逐 token logprob，而非直接传 labels 给 cross_entropy）：

- `chosen_input_ids: [T]`
- `x_chosen = input[:-1]`: `[T-1]`
- `y_chosen = input[1:]`: `[T-1]`
- `mask_chosen = loss_mask[1:]`: `[T-1]`
  - 为什么右移 1？
  - 因为 DPO 的 `logits_to_log_probs` 会在训练里用 `labels` 对齐到“预测 token”，即：
    - logits 位置 `t` 对应 label 位置 `t+1`（自回归 shift）。

DataLoader batch 之后的形状：

- `x_chosen/x_rejected: [B, T-1]`
- `y_chosen/y_rejected: [B, T-1]`
- `mask_chosen/mask_rejected: [B, T-1]`

这些张量在 `train_dpo.py` 的 `dpo_loss(...)` 内部通过 `mask` 做“assistant token 的求和平均”。

---

## 7. 奖励对齐数据：`RLAIFDataset`（prompt/answer 字符串）

`RLAIFDataset` 的输出更“轻量”：它不返回 token 张量，而是返回模型需要继续生成的 `prompt`（字符串）以及 `answer`（这里固定为空字符串）。

### 7.1 init（逐行）

```python
195: class RLAIFDataset(Dataset):
196:     def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
197:         super().__init__()
198:         self.tokenizer = tokenizer
199:         self.max_length = max_length
200:         self.thinking_ratio = thinking_ratio
201:         self.samples = load_dataset('json', data_files=jsonl_path, split='train')
202:         self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
203:         self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids
```

- 注意：`bos_id/eos_id` 在此文件里基本没直接用于 `create_chat_prompt`，更多是为模板或后续扩展保留。
- `thinking_ratio` 决定每条样本是否开启 `open_thinking`（模板里会决定 `<think>...</think>` 是否写出来）。

### 7.2 create_chat_prompt（逐行+动机）

```python
208: def create_chat_prompt(self, conversations):
209:     conversations = pre_processing_chat(conversations)
210:     use_thinking = random.random() < self.thinking_ratio
211:     return self.tokenizer.apply_chat_template(
212:         conversations[:-1],
213:         tokenize=False,
214:         open_thinking=use_thinking,
215:         add_generation_prompt=True
216:     )
```

语义：

- `conversations[:-1]`：丢掉最后一条消息，通常最后一条是“需要模型回答的 user 问题”。也就是说，prompt 只包含到“要回答之前”的上下文。
- `add_generation_prompt=True`：模板会追加 assistant 开头段（并根据 `open_thinking` 决定是否给出 `<think>` 标签）。

### 7.3 __getitem__：返回字符串

```python
217: def __getitem__(self, index):
218:     sample = self.samples[index]
219:     prompt = self.create_chat_prompt(sample['conversations'])
221:     return {'prompt': prompt, 'answer': ""}
```

在 `train_ppo.py` / `train_grpo.py` 里：

- DataLoader batch 后 `prompt` 会是 `list[str]`（训练脚本里再调用 tokenizer 把 prompt token 化）。

---

## 8. 工具调用 RL：`AgentRLDataset`（messages/tools/gt）

Agent RL 训练比 GRPO/PPO 更复杂，它需要：

- prompt 构造时携带可用工具的签名（`tools`）
- 奖励计算时需要与 ground-truth（`gt`）进行对齐/验证

### 8.1 init：逐行读取 jsonl（逐行）

```python
226: class AgentRLDataset(Dataset):
227:     def __init__(self, jsonl_path, tokenizer, max_length=1024):
228:         super().__init__()
229:         self.tokenizer = tokenizer
230:         self.max_length = max_length
231:         self.samples = []
232:         with open(jsonl_path, 'r', encoding='utf-8') as f:
233:             for line in f:
234:                 self.samples.append(json.loads(line.strip()))
```

与其它 Dataset 不同，这里没有用 HF `load_dataset`，而是手工读文件行并 `json.loads`。

### 8.2 parse_conversations：抽取 tools + 截断最后一轮（逐行）

```python
239: def parse_conversations(self, conversations):
240:     messages = []
241:     tools = None
242:     for message in conversations:
243:         message = dict(message)
244:         if message.get("role") == "system" and message.get("tools"):
245:             tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
246:         messages.append(message)
247:     return messages[:-1], tools
```

语义：

- `messages[:-1]`：保留到倒数第二条消息，即把最后一条当作“当前需要回答/需要开始工具交互的 user 问题”而不放入 context。
- `tools`：从 system message 的 `tools` 字段解析出工具签名（可能是 json string 或已是对象）。

### 8.3 __getitem__：返回 messages/tools/gt（逐行+结构）

```python
249: def __getitem__(self, index):
250:     sample = self.samples[index]
251:     messages, tools = self.parse_conversations(sample['conversations'])
252:     return {'messages': messages, 'tools': tools, 'gt': sample['gt']}
```

结构在 `train_agent.py` 中的用法：

- `collate_fn` 把 batch 内每项收集成：
  - `messages: list[list[dict]]`
  - `tools: list[tools_obj]`
  - `gt: list[...]`
- rollout / reward 函数会把 `gt` 与 tool_call 执行结果进行验证与奖励打分。

---

## 9. 本章小结（已覆盖全部 Dataset 类）

你现在应该掌握三种数据语义，它们分别对应三类训练：

1. `PretrainDataset`：`labels=input_ids`，pad -> `-100`；纯 next-token 语言建模。
2. `SFTDataset`：用 `bos_id/eos_id` 定位 assistant 片段，只对 assistant token 位置写入 `labels`；其余位置为 `-100`。
3. `DPODataset`：chosen/rejected 两路，并通过 `generate_loss_mask` 得到 assistant 片段 mask；输出 `x/y/mask` 对齐自回归 shift（mask 也右移 1）。
4. `RLAIFDataset`：输出 prompt 字符串（可能启用 open_thinking），训练脚本再 token 化并生成 completion。
5. `AgentRLDataset`：输出多轮 agent 所需的 `messages/tools/gt`，用于 rollout 的 tool call 解析与 reward 验证。

下一章 `04_训练基础设施（分布式/检查点/采样器）` 我会逐行解析 `trainer/trainer_utils.py`，重点把 checkpoint 的保存/恢复字段与张量 dtype（half/AMP）这两件“工程细节”说清楚。

