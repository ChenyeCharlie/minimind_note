# 01_核心模型 MiniMind（逐行+张量维度）

本章聚焦 `minimind_src/model/model_minimind.py` 的核心实现。要求你后续每一处张量维度变化都能从这里“对上号”，所以本章会对关键维度做显式推导（尤其是 Attention、RoPE、MoE、KV cache 与 CausalLM loss shift）。

---

## 0. 文件入口：`model_minimind.py` 的角色

该文件定义了：

- `MiniMindConfig`：模型超参与 MoE/RoPE 配置载体
- `RMSNorm`：RMSNorm 实现
- RoPE：`precompute_freqs_cis`、`apply_rotary_pos_emb`
- GQA/重复 KV：`repeat_kv`
- Transformer 子层：`Attention`、`FeedForward`、`MOEFeedForward`、`MiniMindBlock`
- 堆叠：`MiniMindModel`
- CausalLM 封装与采样：`MiniMindForCausalLM`

整个前向与生成都围绕同一套维度骨架：`[B, T, C]` -> `logits [B, T, V]`。

---

## 1. Import 与依赖（逐行）

```python
1: import math, torch, torch.nn.functional as F
2: from torch import nn
3: from transformers.activations import ACT2FN
4: from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
5: from transformers.modeling_outputs import MoeCausalLMOutputWithPast
```

- 第 1 行：`math` 用于 `sqrt`、余弦/对数缩放等；`F` 用于 softmax、scaled dot-product 等函数型算子。
- 第 2 行：`nn` 用于声明模块（Linear/Dropout/Embedding 等）。
- 第 3 行：`ACT2FN` 把字符串激活名映射到对应激活函数（如 `silu`）。
- 第 4 行：`PreTrainedModel/GenerationMixin/PretrainedConfig` 让该模型能对接 Transformers 的 generate 生态与配置管理。
- 第 5 行：`MoeCausalLMOutputWithPast` 是标准输出容器（包含 `loss/aux_loss/logits/past_key_values/...`）。

---

## 2. `MiniMindConfig`（配置逐行）

```python
10: class MiniMindConfig(PretrainedConfig):
11:     model_type = "minimind"
12:     def __init__(self, hidden_size=768, num_hidden_layers=8, use_moe=False, **kwargs):
13:         super().__init__(**kwargs)
14:         self.hidden_size = hidden_size
15:         self.num_hidden_layers = num_hidden_layers
16:         self.use_moe = use_moe
17:         self.dropout = kwargs.get("dropout", 0.0)
18:         self.vocab_size = kwargs.get("vocab_size", 6400)
19:         self.bos_token_id = kwargs.get("bos_token_id", 1)
20:         self.eos_token_id = kwargs.get("eos_token_id", 2)
21:         self.flash_attn = kwargs.get("flash_attn", True)
22:         self.num_attention_heads = kwargs.get("num_attention_heads", 8)
23:         self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
24:         self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
25:         self.hidden_act = kwargs.get("hidden_act", 'silu')
26:         self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
27:         self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)
28:         self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
29:         self.rope_theta = kwargs.get("rope_theta", 1e6)
30:         self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)
31:         self.rope_scaling = {
32:             "beta_fast": 32,
33:             "beta_slow": 1,
34:             "factor": 16,
35:             "original_max_position_embeddings": 2048,
36:             "attention_factor": 1.0,
37:             "type": "yarn"
38:         } if self.inference_rope_scaling else None
39:         ### MoE specific configs (ignored if use_moe = False)
40:         self.num_experts = kwargs.get("num_experts", 4)
41:         self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)
42:         self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)
43:         self.norm_topk_prob = kwargs.get("norm_topk_prob", True)
44:         self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)
```

关键点（你后面维度会反复用到）：

- `hidden_size=C`：输入/输出通道维度。
- `num_attention_heads=n_heads`、`head_dim=d`：
  - 期望满足 `C = n_heads * head_dim`（这里默认 `head_dim = C // n_heads`）。
- `num_key_value_heads=n_kv_heads`：
  - 支持 GQA：`n_kv_heads <= n_heads`，并通过 `repeat_kv` 把 k/v 扩展回 `n_heads`。
- MoE：
  - `use_moe` 控制每个 Block 使用 `FeedForward`（dense）还是 `MOEFeedForward`（MoE）。
  - `num_experts`：专家数量。
  - `num_experts_per_tok=k`：每个 token 选择 Top-k 个专家（当前 forward 里用 `torch.topk(..., k=k)`）。
- RoPE：
  - `max_position_embeddings` 决定预计算频率的长度（buffer 长度）。
  - `inference_rope_scaling` 若启用，则 `rope_scaling` 被设成 YaRN 风格参数集合，并影响 `precompute_freqs_cis` 里频率 `freqs` 的缩放。

---

## 3. `RMSNorm`（逐行+维度）

```python
49: class RMSNorm(torch.nn.Module):
50:     def __init__(self, dim: int, eps: float = 1e-5):
51:         super().__init__()
52:         self.eps = eps
53:         self.weight = nn.Parameter(torch.ones(dim))

55:     def norm(self, x):
56:         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

58:     def forward(self, x):
59:         return (self.weight * self.norm(x.float())).type_as(x)
```

维度追踪：

- 假设输入 `x` 为 `[..., dim]`（Attention 里会是 `[B,T,n_heads,head_dim]` 或 `[B,T,C]` 这类）。
- 第 56 行：`x.pow(2).mean(-1, keepdim=True)` 的 `-1` 维是 `dim`，所以：
  - `[..., dim] -> [..., 1]`
  - 再 `rsqrt` 得到同形 `[..., 1]`，与 `x` 广播相乘，仍是 `[..., dim]`。

数值/动机：

- RMSNorm 不减均值，仅对二阶矩做归一，因此更轻且在一些 LLM 实现里更稳定。
- 第 59 行把 norm 的计算转成 `float()` 再乘回原 dtype：这是常见的数值稳定策略（避免低精度溢出）。

---

## 4. RoPE 预计算：`precompute_freqs_cis`（逐行+维度）

```python
61: def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
62:     freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
63:     if rope_scaling is not None: # YaRN: ...
64:         orig_max, factor, beta_fast, beta_slow, attn_factor = (...)
68:         if end / orig_max > 1.0:
69:             inv_dim = lambda b: ...
70:             low, high = ...
71:             ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
72:             freqs = freqs * (1 - ramp + ramp / factor)
73:     t = torch.arange(end, device=freqs.device)
74:     freqs = torch.outer(t, freqs).float()
75:     freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
76:     freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
77:     return freqs_cos, freqs_sin
```

维度追踪（假设 `dim=head_dim=d`）：

1. 第 62 行：
   - `torch.arange(0, dim, 2)[:(dim//2)]` 长度为 `d/2`
   - 所以 `freqs` 初始形状为 `[d/2]`。
2. 第 73-74 行：
   - `t = [end]`
   - `freqs = outer(t, freqs)` -> `[end, d/2]`
3. 第 75-76 行：
   - `torch.cos(freqs)` -> `[end, d/2]`
   - `torch.cat([... , ...], dim=-1)` -> `[end, d]`
   - 同理 `freqs_sin` -> `[end, d]`
4. 返回：
   - `freqs_cos: [end, d]`
   - `freqs_sin: [end, d]`

动机：

- RoPE 用角度频率对不同维度注入相对位置信息；`rope_scaling`（YaRN）是在做“外推到更长上下文”时的频率调制。

---

## 5. RoPE 应用：`apply_rotary_pos_emb`（逐行+广播规则）

```python
79: def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
80:     def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
81:     q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
82:     k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
83:     return q_embed, k_embed
```

这里的核心是广播维度对齐：

- 注意 Attention 里传入：
  - `q: [B, seq_len, n_heads, head_dim]`
  - `k: [B, seq_len, n_kv_heads, head_dim]`
  - `cos, sin` 来自 `MiniMindModel` 的 buffer 切片：
    - `cos: [seq_len, head_dim]`
    - `sin: [seq_len, head_dim]`
- 第 81 行：
  - `cos.unsqueeze(unsqueeze_dim)` 默认 `unsqueeze_dim=1`：
    - `cos [seq_len, d] -> [seq_len, 1, d]`
  - 与 `q [B, seq_len, n_heads, d]` 广播：
    - 最后一维 `d` 匹配
    - 第二个维度 `1` 可广播到 `n_heads`
    - `seq_len` 对齐到 q 的第二维
  - 得到 `q_embed` 仍是 `[B, seq_len, n_heads, d]`
- `rotate_half`：
  - 把最后一维按一半拆成两块并进行符号与交换：这是标准 RoPE 旋转实现的向量形式。

动机：

- RoPE 让模型在计算注意力时获得相对位置信息，而不依赖绝对位置 embedding。

---

## 6. GQA 的 KV 重复：`repeat_kv`（逐行+维度）

```python
85: def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
86:     bs, slen, num_key_value_heads, head_dim = x.shape
87:     if n_rep == 1: return x
88:     return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```

输入维度：

- `x: [B, T, n_kv_heads, d]`
- `n_rep = n_heads / n_kv_heads`

输出维度（当 `n_rep>1`）：

- `expand` 后：`[B, T, n_kv_heads, n_rep, d]`
- `reshape`：`[B, T, n_kv_heads*n_rep, d] = [B, T, n_heads, d]`

动机：

- GQA 情况下 kv head 数更少以省显存/算力；注意力计算需要 q head 数一致，所以用重复策略对齐。

---

## 7. `Attention`（逐行+张量维度完整追踪）

### 7.1 初始化（逐行）

```python
90: class Attention(nn.Module):
91:     def __init__(self, config: MiniMindConfig):
92:         super().__init__()
93:         self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
94:         self.n_local_heads = config.num_attention_heads
95:         self.n_local_kv_heads = self.num_key_value_heads
96:         self.n_rep = self.n_local_heads // self.n_local_kv_heads
97:         self.head_dim = config.head_dim
98:         self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
99:         self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
100:        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
101:        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
102:        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
103:        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
104:        self.attn_dropout = nn.Dropout(config.dropout)
105:        self.resid_dropout = nn.Dropout(config.dropout)
106:        self.dropout = config.dropout
107:        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
```

关键维度关系：

- `q_proj` 输出维度：`n_heads * d`
- `k_proj/v_proj` 输出维度：`n_kv_heads * d`
- `o_proj` 从 `n_heads*d (= hidden_size)` 回到 `hidden_size`。

额外动机：

- `q_norm/k_norm` 在 RoPE 前对 q/k 做 RMSNorm，有助于稳定注意力打分尺度。
- `flash` 分支：使用 PyTorch 原生 `scaled_dot_product_attention`（条件满足时）。

### 7.2 前向（逐行+维度）

```python
109:    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
110:        bsz, seq_len, _ = x.shape
111:        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
112:        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
113:        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
114:        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
115:        xq, xk = self.q_norm(xq), self.k_norm(xk)
116:        cos, sin = position_embeddings
117:        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
118:        if past_key_value is not None:
119:            xk = torch.cat([past_key_value[0], xk], dim=1)
120:            xv = torch.cat([past_key_value[1], xv], dim=1)
121:        past_kv = (xk, xv) if use_cache else None
122:        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
```

先定义维度符号：

- 输入 `x`: `[B, T_new, C]`
- `C = n_heads * d`
- `T_new`：当前前向输入的序列长度（生成时可能是 1）

逐行维度：

- 第 110 行：`bsz=B, seq_len=T_new`。
- 第 111 行：
  - `xq = q_proj(x)`: `[B, T_new, n_heads*d]`
  - `xk = k_proj(x)`: `[B, T_new, n_kv_heads*d]`
  - `xv = v_proj(x)`: `[B, T_new, n_kv_heads*d]`
- 第 112-114 行 `view`：
  - `xq: [B, T_new, n_heads, d]`
  - `xk: [B, T_new, n_kv_heads, d]`
  - `xv: [B, T_new, n_kv_heads, d]`
- 第 115 行：
  - `q_norm/k_norm` 不改变 shape：仍是上述 `[... , d]`
- 第 117 行 RoPE：
  - `cos/sin: [T_new, d]`（从 buffer 切片而来）
  - `apply_rotary_pos_emb` 广播后输出：
    - `xq: [B, T_new, n_heads, d]`
    - `xk: [B, T_new, n_kv_heads, d]`
- KV cache（第 118-120 行）：
  - 当 `past_key_value is not None`，`past_key_value[0]` 在该实现中保存的是 `xk` 的原始未 repeat 形状：
    - `past_key_value[0]: [B, T_past, n_kv_heads, d]`
    - `past_key_value[1]: [B, T_past, n_kv_heads, d]`
  - cat dim=1：
    - `xk: [B, T_past + T_new, n_kv_heads, d]`
    - `xv: [B, T_past + T_new, n_kv_heads, d]`
- 第 121 行 cache 输出：
  - `past_kv=(xk,xv)`：用于下一轮生成，形状为：
    - `past_k: [B, T_total, n_kv_heads, d]`
    - `past_v: [B, T_total, n_kv_heads, d]`
- 第 122 行：准备 attention 计算
  - `xq.transpose(1,2)`：`[B, n_heads, T_new, d]`
  - `repeat_kv(xk, n_rep)`：`[B, T_total, n_heads, d]`，再 transpose -> `[B, n_heads, T_total, d]`
  - `xv` 同理 -> `[B, n_heads, T_total, d]`

接着是 attention 打分两条分支：

```python
123:        if self.flash and (seq_len > 1) and (past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
124:            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
125:        else:
126:            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
127:            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
128:            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
129:            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv
130:        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
131:        output = self.resid_dropout(self.o_proj(output))
132:        return output, past_kv
```

### 7.2.1 Flash 分支（条件解释）

- 条件要求：
  - `self.flash` 存在原生 SDPA
  - `seq_len > 1` 且 `past_key_value is None`（即一次性整段 prompt，而不是增量生成）
  - `attention_mask is None or attention_mask 全 1`（无 padding）
- 动机：SDPA 对这类输入有更高效内核。

`scaled_dot_product_attention` 输入维度期望与我们一致：

- `xq: [B, n_heads, T_new, d]`
- `xk: [B, n_heads, T_total, d]`
- `xv: [B, n_heads, T_total, d]`

输出仍为：

- `output: [B, n_heads, T_new, d]`（PyTorch 保持最后两维一致）

### 7.2.2 非 Flash 分支（scores/softmax/matmul）

- 第 126 行：
  - `xq @ xk.transpose(-2,-1)`
    - `xq: [B, n_heads, T_new, d]`
    - `xk.transpose(-2,-1): [B, n_heads, d, T_total]`
    - scores: `[B, n_heads, T_new, T_total]`
  - 除以 `sqrt(head_dim)`：
    - 动机：控制点积方差，避免 softmax 饱和。
- 第 127 行 causal mask：
  - 只对“最后 `seq_len` 的 query 对应的 k 段”加上上三角 `-inf`（实现方式是对 `scores[..., -seq_len:]` 的最后块做三角掩码）。
  - `triu(1)` 把对角线以上置 `-inf`，保证自回归不看未来。
- 第 128 行 padding mask：
  - `attention_mask: [B, T_total]`（生成时它会随时间增长）
  - `attention_mask.unsqueeze(1).unsqueeze(2)` -> `[B, 1, 1, T_total]`
  - `1.0 - mask` 为 1 的地方加 `-1e9` 到 scores，softmax 后概率接近 0。
- 第 129 行：
  - softmax over最后维：仍 `[B, n_heads, T_new, T_total]`
  - 与 `xv [B, n_heads, T_total, d]` 相乘 -> `[B, n_heads, T_new, d]`
- 第 130 行：
  - `output.transpose(1,2)`：`[B, T_new, n_heads, d]`
  - reshape 成 `[B, T_new, n_heads*d] = [B, T_new, C]`
- 第 131 行：
  - `o_proj`：`[B, T_new, C] -> [B, T_new, C]`
  - resid_dropout：不改变维度

返回值：

- `output`: `[B, T_new, C]`
- `past_kv`（如果 use_cache=True）：
  - `(k: [B, T_total, n_kv_heads, d], v: [B, T_total, n_kv_heads, d])`
  - 否则为 `None`

---

## 8. `FeedForward`（逐行+维度）

```python
134: class FeedForward(nn.Module):
135:     def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
136:         super().__init__()
137:         intermediate_size = intermediate_size or config.intermediate_size
138:         self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
139:         self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
140:         self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
141:         self.act_fn = ACT2FN[config.hidden_act]

143:     def forward(self, x):
144:         return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

维度：

- 输入 `x: [B, T, C]`
- gate_proj/up_proj：
  - `[B,T,C] -> [B,T,intermediate]`
- SwiGLU/GLU 风格：
  - `act(gate_proj(x)) * up_proj(x)`：仍 `[B,T,intermediate]`
- down_proj：
  - `[B,T,intermediate] -> [B,T,C]`

动机：

- 这种“门控激活”（GLU/SwiGLU）通常比简单 GELU 更擅长表达。

---

## 9. `MOEFeedForward`（逐行+维度与路由）

```python
146: class MOEFeedForward(nn.Module):
147:     def __init__(self, config: MiniMindConfig):
148:         super().__init__()
149:         self.config = config
150:         self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
151:         self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
152:         self.act_fn = ACT2FN[config.hidden_act]

154:     def forward(self, x):
155:         batch_size, seq_len, hidden_dim = x.shape
156:         x_flat = x.view(-1, hidden_dim)
157:         scores = F.softmax(self.gate(x_flat), dim=-1)
158:         topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
159:         if self.config.norm_topk_prob: topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
160:         y = torch.zeros_like(x_flat)
161:         for i, expert in enumerate(self.experts):
162:             mask = (topk_idx == i)
163:             if mask.any():
164:                 token_idx = mask.any(dim=-1).nonzero().flatten()
165:                 weight = topk_weight[mask].view(-1, 1)
166:                 y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
167:             elif self.training:
168:                 y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
169:         if self.training and self.config.router_aux_loss_coef > 0:
170:             load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
171:             self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
172:         else:
173:             self.aux_loss = scores.new_zeros(1).squeeze()
174:         return y.view(batch_size, seq_len, hidden_dim)
```

维度追踪：

- 输入 `x: [B, T, C]`
- `batch_size=B, seq_len=T, hidden_dim=C`
- `x_flat = x.view(-1, hidden_dim)`：
  - `[B,T,C] -> [B*T, C]`
- `scores = softmax(gate(x_flat))`：
  - `gate: C -> num_experts`
  - 所以 `scores: [B*T, E]`
- `topk_weight/topk_idx = topk(scores, k=k, dim=-1)`：
  - `topk_weight: [B*T, k]`
  - `topk_idx: [B*T, k]`
- `y = zeros_like(x_flat)` -> `[B*T, C]`
- 循环每个 expert：
  - `mask = (topk_idx == i)`：`[B*T, k]` 布尔
  - `token_idx = mask.any(dim=-1).nonzero().flatten()`：
    - `mask.any(dim=-1)` -> `[B*T]`
    - nonzero -> 选择 token 的索引，形状 `[N_i]`
  - `expert(x_flat[token_idx])` -> `[N_i, C]`
  - `weight = topk_weight[mask].view(-1, 1)`：
    - `topk_weight[mask]` 会摊平所有 True 的位置（通常是与该 expert 被选中的 token-slot 数一致）
    - 最终 `weight: [N_i, 1]`（若每个 token 至多一次命中该 expert）
  - `expert(x_flat[token_idx]) * weight` -> `[N_i, C]`
  - `y.index_add_(0, token_idx, ...)`：把这些 token 的输出按索引加回到 `y` 中。
- 输出 reshape：`y.view(B,T,C) -> [B,T,C]`

动机与注意点：

- `norm_topk_prob`：对每个 token 选出来的 k 个专家概率重新归一，防止 topk 概率和影响幅度。
- `aux_loss`：
  - 第 170 行：
    - `topk_idx: [B*T, k]`
    - `F.one_hot(topk_idx, E)`：对每个 token-slot 的专家 id 做 one-hot，得到 `[B*T, k, E]`
    - `.float().mean(0)`：沿第 0 维（把所有 token-slot 样本平均）得到 `load: [k, E]`
    - 直观含义：第 `slot`（top-k 中第几名）层面上，各专家被选中的“平均负载”
  - 第 171 行：
    - `scores: [B*T, E]`
    - `scores.mean(0)`: ` [E]`（所有 token 的平均路由概率）
    - `load * scores.mean(0)`：`[k, E] * [E]` 广播到 `[k, E]`
    - `.sum()`：得到标量
    - 再乘上 `self.config.num_experts * self.config.router_aux_loss_coef`，把该正则项按专家数与系数缩放
  - 该 `aux_loss` 会在 `MiniMindModel.forward` 中被收集（只收集 `MOEFeedForward` 层的 `l.mlp.aux_loss`）并加到训练总损失里（见 `res.loss + res.aux_loss`）。

---

## 10. `MiniMindBlock`（逐行+残差结构）

```python
176: class MiniMindBlock(nn.Module):
177:     def __init__(self, layer_id: int, config: MiniMindConfig):
178:         super().__init__()
179:         self.self_attn = Attention(config)
180:         self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
181:         self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
182:         self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

184:     def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
185:         residual = hidden_states
186:         hidden_states, present_key_value = self.self_attn(
187:             self.input_layernorm(hidden_states), position_embeddings,
188:             past_key_value, use_cache, attention_mask
189:         )
190:         hidden_states += residual
191:         hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
192:         return hidden_states, present_key_value
```

结构含义：

1. Pre-LN Attention：
   - 先 `RMSNorm`，再送入 attention。
2. Residual add：
   - attention 输出 `[B,T,C]` 与 residual `[B,T,C]` 相加。
3. MLP（Dense 或 MoE）：
   - 再做一次 RMSNorm，然后进 MLP，再与当前 hidden_states 相加。

维度不变性：

- 注意这里整个 Block 的输入输出维度恒为 `[B,T,C]`。

---

## 11. `MiniMindModel`（堆叠+KV cache+aux_loss 聚合）

```python
194: class MiniMindModel(nn.Module):
195:     def __init__(self, config: MiniMindConfig):
196:         super().__init__()
197:         self.config = config
198:         self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
199:         self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
200:         self.dropout = nn.Dropout(config.dropout)
201:         self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
202:         self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
203:         freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
204:         self.register_buffer("freqs_cos", freqs_cos, persistent=False)
205:         self.register_buffer("freqs_sin", freqs_sin, persistent=False)

207:     def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
208:         batch_size, seq_length = input_ids.shape
209:         if hasattr(past_key_values, 'layers'): past_key_values = None
210:         past_key_values = past_key_values or [None] * len(self.layers)
211:         start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
212:         hidden_states = self.dropout(self.embed_tokens(input_ids))
213:         position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
214:         presents = []
215:         for layer, past_key_value in zip(self.layers, past_key_values):
216:             hidden_states, present = layer(
217:                 hidden_states,
218:                 position_embeddings,
219:                 past_key_value=past_key_value,
220:                 use_cache=use_cache,
221:                 attention_mask=attention_mask
222:             )
223:             presents.append(present)
224:         hidden_states = self.norm(hidden_states)
225:         aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
226:         return hidden_states, presents, aux_loss
```

维度追踪：

- 输入 `input_ids: [B, T]`
- embedding：`embed_tokens` -> `[B, T, C]`
- 每个 block 都保持 `[B,T,C]`
- 最后 norm -> `[B,T,C]`

KV cache 结构：

- `past_key_values` 在外部（generate 或 transformers）传入时，期望是“每层一个 past_kv”列表。
- 这里做了一个鲁棒性判断：
  - `if hasattr(past_key_values, 'layers')`: 可能是某种 transformers 内部结构，直接置为 None。
- `past_key_values = past_key_values or [None] * len(self.layers)`：
  - 保证长度与层数一致。
- `start_pos`：
  - 取第 0 层 past_k 的第 1 维长度作为“历史累计长度 `T_past`”。
  - 由于 Attention 的 cache 未 transpose，所以 `past_key_values[0][0].shape[1]` 是 `T_total` 中的历史部分。
- `position_embeddings`：
  - 从 buffer 中切片长度为 `seq_length=T_new` 的 cos/sin。

aux_loss 聚合：

- 仅对 `isinstance(l.mlp, MOEFeedForward)` 的层取 `l.mlp.aux_loss`。
- dense 层没有 `aux_loss` 属性（但也不会进入列表）。
- `sum(..., hidden_states.new_zeros(1).squeeze())`：
  - 以当前 dtype/device 的 0 标量作为初始值，避免 device mismatch。

---

## 12. `MiniMindForCausalLM`：LM head、loss shift、generate（逐行+维度）

### 12.1 init（逐行）

```python
228: class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
229:     config_class = MiniMindConfig
230:     def __init__(self, config: MiniMindConfig = None):
231:         self.config = config or MiniMindConfig()
232:         super().__init__(self.config)
233:         self.model = MiniMindModel(self.config)
234:         self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
235:         self.model.embed_tokens.weight = self.lm_head.weight
```

要点：

- 通过 `weight tying`：embedding 与 lm_head 共享权重矩阵，这能减少参数并常见于语言模型。

### 12.2 forward（逐行+loss 对齐）

```python
237:     def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
238:         hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
239:         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
240:         logits = self.lm_head(hidden_states[:, slice_indices, :])
241:         loss = None
242:         if labels is not None:
243:             x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
244:             loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
245:         return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

维度追踪：

- `self.model` 输出：
  - `hidden_states: [B, T, C]`
  - `logits` 前会对 `hidden_states[:, slice_indices, :]` 做切片（如只算最后 n_keep tokens）。
- lm_head：
  - `[B, T_keep, C] -> [B, T_keep, V]`
- loss shift：
  - `logits[..., :-1, :]`：`[B, T_keep-1, V]`
  - `labels[..., 1:]`：`[B, T_keep-1]`
  - `view(-1, V)`：`[(B*(T_keep-1)), V]`
  - `y.view(-1)`：`[B*(T_keep-1)]`
- `ignore_index=-100`：
  - dataset 里用 `-100` 把不需要参与 loss 的位置屏蔽掉。

### 12.3 generate（采样循环逐行+维度）

```python
249:     @torch.inference_mode()
250:     def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
250:         input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
251:         attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
252:         past_key_values = kwargs.pop("past_key_values", None)
253:         finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
254:         if streamer: streamer.put(input_ids.cpu())
255:         for _ in range(max_new_tokens):
256:             past_len = past_key_values[0][0].shape[1] if past_key_values else 0
257:             outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
258:             attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
259:             logits = outputs.logits[:, -1, :] / temperature
260:             if repetition_penalty != 1.0:
261:                 for i in range(input_ids.shape[0]): logits[i, torch.unique(input_ids[i])] /= repetition_penalty
262:             if top_k > 0: 
263:                 logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
264:             if top_p < 1.0:
265:                 sorted_logits, sorted_indices = torch.sort(logits, descending=True)
266:                 mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
267:                 mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
268:                 logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
269:             next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
270:             if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
271:             input_ids = torch.cat([input_ids, next_token], dim=-1)
272:             past_key_values = outputs.past_key_values if use_cache else None
273:             if streamer: streamer.put(next_token.cpu())
274:             if eos_token_id is not None:
275:                 finished |= next_token.squeeze(-1).eq(eos_token_id)
276:                 if finished.all(): break
277:         if streamer: streamer.end()
278:         if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
279:         return input_ids
```

维度追踪（以第 `t` 轮为例）：

- 假设初始 prompt 长度为 `P`，batch size 为 `B0`。
- repeat 后：`input_ids: [B0*num_return_sequences, P]`
- 第 `_` 轮开始时：
  - 经过 `t` 次追加后，`input_ids` 长度为 `P+t`
  - `past_len` 表示 cache 的历史长度（首次为 0，之后通常为 `P+t-1` 或 `P+...`，取决于实现是否缓存 prompt 全量）
- 第 257 行：
  - `input_ids[:, past_len:]` 只取增量 token：
    - shape: `[B, (P+t) - past_len]`（第一次通常为 `[B, P]`，之后通常为 `[B, 1]`）
  - `outputs.logits` 会对应增量长度的 logits，因此：
    - `outputs.logits[:, -1, :]` 取当前步最后一个 token 的分布：`[B, V]`
- 第 258 行：
  - `attention_mask` 在非 None 时追加一个 `1`，确保下一步 attention mask 与总长度对齐。

采样循环中的逐行关键逻辑与维度（在每一轮生成中都重复）：

1. `logits = outputs.logits[:, -1, :] / temperature`
   - `outputs.logits` 在 CausalLM 中形如 `[B, T_keep, V]`
   - `[:, -1, :]` 取最后一步 token 的分布：`logits: [B, V]`
   - `/ temperature`：改变分布“尖/平”的尺度（temperature 越小越贪心）

2. `repetition_penalty != 1.0` 分支
   - 先 `for i in range(input_ids.shape[0])`：逐样本处理
   - `torch.unique(input_ids[i])`：找出当前序列中已经出现过的 token id（形状 `[K]`）
   - `logits[i, unique_ids] /= repetition_penalty`
   - 维度仍然是对 `logits: [B, V]` 的子索引就地修改，不改变 logits 全形

3. `top_k > 0`
   - `torch.topk(logits, top_k)[0]`：取 top-k 的值，形状 `[B, top_k]`
   - `[..., -1, None]`：取第 k 个阈值并扩展维度为 `[B, 1]`
   - `logits[logits < threshold] = -inf`：把低于阈值的 token 彻底禁止

4. `top_p < 1.0`（nucleus）
   - `sorted_logits, sorted_indices = torch.sort(logits, descending=True)`
     - `sorted_logits: [B, V]`
     - `sorted_indices: [B, V]`
   - `mask = cumsum(softmax(sorted_logits)) > top_p`
     - `softmax(sorted_logits)`: `[B, V]`
     - `cumsum(...)`: `[B, V]`
     - `mask`: bool `[B, V]`
   - `mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0`
     - 让“累计概率第一次超过阈值的 token”也仍保留，从而至少保留一个 token（避免全 -inf）
   - `logits[mask.scatter(1, sorted_indices, mask)] = -inf`
     - `scatter` 把在排序空间中的 mask 映射回原 vocab id 空间

5. 采样得到 `next_token`
   - `do_sample=True`：
     - `softmax(logits)`: `[B, V]`
     - `torch.multinomial(..., num_samples=1)`: `next_token: [B, 1]`
   - `do_sample=False`：
     - `torch.argmax(logits, dim=-1, keepdim=True)`: `next_token: [B, 1]`

6. EOS 强制策略
   - `finished` 初始：`[B]`，bool
   - `finished.unsqueeze(-1)`: `[B, 1]`
   - `torch.where(finished.unsqueeze(-1), next_token.new_full((B,1), eos_token_id), next_token)`
     - 输出仍是 `[B, 1]`

7. 序列增长与 KV cache 更新
   - `input_ids = torch.cat([input_ids, next_token], dim=-1)`
     - `input_ids` 在循环中逐步增长：`[B, cur_len] -> [B, cur_len+1]`
   - `past_key_values = outputs.past_key_values if use_cache else None`
     - 若 `use_cache=True`，则下一轮 `past_len` 会从 `past_key_values[0][0].shape[1]` 读取

8. 流式输出与停止
   - `if streamer: streamer.put(next_token.cpu())`：把本步 token 推送到输出流
   - `finished |= next_token.squeeze(-1).eq(eos_token_id)`
     - `next_token.squeeze(-1)`: `[B]`
     - `eq(eos_token_id)`: `[B]`
   - `if finished.all(): break`：所有样本都结束则提前停止

停止条件总结：

- EOS 触发由 `finished` 维护；一旦 `finished.all()` 为 True，就退出 `for _ in range(max_new_tokens)`。

---

## 13. 本章小结（你应当掌握的“可复核点”）

1. 注意力的标准维度流（non-flash 分支）：
   - `x [B,T,C]`
   - `q [B,T,n_heads,d] -> transpose -> [B,n_heads,T,d]`
   - `k/v [B,T,n_kv_heads,d] -> repeat_kv -> [B,T,n_heads,d] -> transpose -> [B,n_heads,T,d]`
   - `scores [B,n_heads,T,T_total]`，softmax 后乘 `v` -> `output [B,n_heads,T,d]`
   - `reshape -> [B,T,C]`
2. KV cache 在该实现中的保存形态：
   - `(k,v)` 在 Attention 中返回的是 `[B,T_total,n_kv_heads,d]` 的未 repeat 形状。
3. CausalLM 的 loss shift：
   - `x = logits[..., :-1, :]` 与 `y = labels[..., 1:]` 对齐；`ignore_index=-100` 屏蔽非训练片段。
4. RoPE 的 cos/sin buffer：
   - `precompute_freqs_cis` 输出 `[max_pos, head_dim]`，切片到 `[seq_len, head_dim]` 后用广播注入到 q/k。

下一章 `02_LoRA 适配与合并（逐行）` 会解释如何把这些线性层与 `model_lora.py` 里的 LoRA 分支“挂接”到一起，以及训练/保存/合并时 state_dict 的键组织规则。

