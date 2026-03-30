# 14_minimind-3/模型工件与 tokenizer/chat_template 解析（单独章）

本章聚焦 `minimind_src/minimind-3/` 这一套“模型工件”（tokenizer/chat_template/config），因为它们决定了：

1. `tokenizer.apply_chat_template(...)` 在训练/推理时到底生成怎样的字符串协议
2. 特殊 token（`<|im_start|> / <|im_end|> / <think> / <tool_call> / <tool_response>` 等）对应的词表 id 与 tokenizer 行为
3. 模型的 RoPE 参数（`rope_theta` 等）和结构维度（hidden_size、heads、head_dim、vocab_size）如何在推理侧生效

涉及文件：

- `minimind_src/minimind-3/chat_template.jinja`
- `minimind_src/minimind-3/tokenizer_config.json`
- `minimind_src/minimind-3/config.json`
- 额外同构字段参考：`minimind_src/model/tokenizer_config.json`

---

## 1. `chat_template.jinja`：对话协议的“唯一来源”

`chat_template.jinja` 是一个 Jinja 模板。`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=..., tools=..., open_thinking=...)` 时，它会把 `messages`（list[dict]）与 `tools`（可选）拼成一段最终文本。

### 1.1 工具模式 vs 非工具模式：`{% if tools %} ... {% else %}`

当 `tools` 非空时，模板会在 system 段插入工具签名区：

```jinja
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call ... <tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>\n\nFor each function call ... <tool_call> ... </tool_call><|im_end|>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
```

这段逻辑对应你在 RL/服务脚本中传入 `tools=tools` 后的行为：

- tokenizer 会强制把可用工具“函数签名列表”显式注入到 system 消息里
- 并给出“返回 `<tool_call>...json...</tool_call>`”的格式约束

当 `tools` 为空时，只输出 system 消息（若 messages[0] 是 system）。

### 1.2 `ns.multi_step_tool`：识别多轮 tool 追踪的停止条件

模板中出现：

```jinja
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" and message.content is string
        and not(message.content.startswith('<tool_response>') and message.content.endswith('</tool_response>')) %}
        {%- set ns.multi_step_tool = false %}
        {%- set ns.last_query_index = index %}
    {%- endif %}
{%- endfor %}
```

尽管当前模板片段中后续没有直接使用 `last_query_index`，但 `multi_step_tool` 的存在意味着：

- 模板设计者预期 messages 中可能存在“tool_response 直接塞回用户内容”的特殊结构
- 当遇到真正的 user 查询时，才认为多步 tool 流程结束（防止把用户查询错误地和 tool 流程混在一起）

### 1.3 拼接 messages：三类 role 分别格式化

遍历 messages 的核心分支：

```jinja
{%- for message in messages %}
...
    {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content.lstrip('\n') }}
        {%- if message.tool_calls %} ... {{- '<tool_call>...</tool_call>' }} ... {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {{- '<|im_start|>user' }}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {{- '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}
```

这段与训练/推理侧的“字符串协议”强绑定，具体含义如下：

1. `user`（以及非首个 system）：
   - 总是用 `<|im_start|>user ... <|im_end|>` 包裹内容
2. `assistant`：
   - 强制输出 `<think>...</think>`（即使为空也会输出占位）
   - 紧接着输出 assistant 的正文 `content`
   - 若 `message.tool_calls` 存在，则追加若干 `<tool_call>...json...</tool_call>`
3. `tool`：
   - 模板把 tool role “转译”为 `<|im_start|>user \n<tool_response> ... </tool_response> <|im_end|>`
   - 也就是说：在模型看来，工具结果被当成一种特殊“用户输入”结构，而不是另一种模型角色（这也解释了训练数据/脚本里对 `<tool_response>` 的特殊处理）

### 1.4 `add_generation_prompt`：为下一轮生成补上 `<|im_start|>assistant` 与 `<think>` 占位

模板末尾：

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if open_thinking is defined and open_thinking is true %}
        {{- '<think>\n' }}
    {%- else %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

对应你代码里常见用法：

- RL rollout/AgentRL：会根据 `open_thinking` 随机决定是否让模型实际输出思考内容
- 当 `open_thinking=False` 时，模板会直接给 `<think>\n\n</think>\n\n` 这种“空思考占位”，使得模型生成从紧跟其后的正文部分开始更稳定

---

## 2. `tokenizer_config.json`：特殊 token 与 tokenizer 的硬规则

### 2.1 BOS/EOS/PAD

从 `minimind_src/minimind-3/tokenizer_config.json` 可读到：

- `bos_token`: `<|im_start|>`
- `eos_token`: `<|im_end|>`
- `pad_token`: `<|endoftext|>`

这会影响脚本里 `model.generate(..., eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)` 的停止/填充行为。

### 2.2 `<tool_call>` / `<tool_response>` / `<think>` 等 token

在 `added_tokens_decoder` 中能看到：

- `<tool_call>` 与 `</tool_call>`（token 类型 `special: false`，但它们仍在词表里占据明确 id）
- `<tool_response>` 与 `</tool_response>`
- `<think>` 与 `</think>`

因此：

- 训练阶段的 `generate_labels` / `loss_mask` 逻辑（例如 DPO/AgentRL）会依赖这些标签“作为可见字符串片段”出现
- 服务阶段的 `parse_response`（`serve_openai_api.py`）与 UI 阶段的渲染（`web_demo.py`）会依赖 `<think>`/`</think>` 的存在来切分 reasoning/content
- 工具调用的解析会依赖 `<tool_call>...</tool_call>` 这两个边界 token

### 2.3 `chat_template` 字段

在 `tokenizer_config.json` 中还存在一个 `chat_template` 字段，它本质上是 `chat_template.jinja` 的“压缩/内嵌版本”（把模板转成字符串存入 tokenizer 配置）。

这也是为什么你可以：

- 直接用 `tokenizer.apply_chat_template(...)`（transformers 内部会使用 chat_template）
- 或用 `convert_model.py` 中的 `convert_json_to_jinja/convert_jinja_to_json` 在两种表示之间切换

---

## 3. `config.json`：结构维度与 RoPE

`minimind_src/minimind-3/config.json`（transformers 格式）给出推理所需的结构参数：

- `hidden_size: 768`
- `num_hidden_layers: 8`
- `num_attention_heads: 8`
- `num_key_value_heads: 4`
- `head_dim: 96`（由 `hidden_size / num_attention_heads` 推导）
- `vocab_size: 6400`
- `rope_theta: 1000000.0`
- `rope_scaling: null`
- `max_position_embeddings: 32768`

与训练侧代码的对应关系（高层映射）：

- `head_dim` 决定 Attention 里 `q/k` 的最后一维大小（通常是 `[B, T, n_heads, head_dim]` 或等价排列）
- `rope_theta` 与 `rope_scaling` 决定 RoPE 的频率尺度；如果 `rope_scaling=null`，则等价于不做额外缩放
- `vocab_size` 决定输出 logits 的最后一维大小：`logits: [B, T, vocab_size]`

---

## 4. 这些工件如何反过来约束训练/RL

把本章“工件”与前面 AgentRL/RL 的代码点名关联一下：

1. AgentRL 与服务脚本都依赖 `<tool_call>...</tool_call>` 的边界
   - `train_agent.py` 的 `parse_tool_calls(new_text)` 用正则提取 tool_call JSON
   - `serve_openai_api.py` / `eval_toolcall.py` 同样按 `<tool_call>` 切分并 `json.loads`
2. completion_mask 需要能定位“模型动作 token vs tool observation token”
   - 在 AgentRL 的 rollout_single 中：生成 token 先写入 `response_mask=1`
   - 工具执行后加入 `messages` 的 tool response 部分，会在下一轮 observe 时落入 `response_mask=0`（不参与 policy loss）
3. `<think>` 标签会影响 reward 规则与 streaming 输出
   - `train_agent.py` 通过 `'</think>'` 是否存在来切分 answer，并给予思考长度/闭合奖励
   - `serve_openai_api.py` 的流式实现会在字符串里寻找 `'</think>'`，把 `<think>` 中内容转成 `reasoning_content` delta

---

## 5. 本章小结

你可以把 `minimind-3/` 的工件理解为三层：

- “协议层”：`chat_template.jinja` 决定模型看到的文本长什么样
- “token 层”：`tokenizer_config.json` 决定特殊标签如何被 tokenized、bos/eos/pad 是什么
- “结构层”：`config.json` 决定注意力头维度与 RoPE 频率，使得 logits 与 RoPE 位置一致

只要这三层一致，后续训练（SFT/DPO/RL/AgentRL）与推理（generate/streaming/tool_call 回填）才会在字符串切分与 token 对齐上保持闭环。

