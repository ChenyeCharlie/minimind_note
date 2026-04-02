# 01_核心模型 MiniMind（逐行+张量维度）

本章聚焦 `minimind_src/model/model_minimind.py` 的核心实现。我们将深入探讨 MiniMind 如何通过 LLaMA 式的架构设计，在极小的参数量下实现完整的语言建模能力。

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

## 1. Import 与依赖：生态对接

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

## 2. `MiniMindConfig`：模型的基因图谱

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
...
22:         self.num_attention_heads = kwargs.get("num_attention_heads", 8)
23:         self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
24:         self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
...
```

关键点：

- `hidden_size=C`：输入/输出通道维度。
- `num_attention_heads=n_heads`、`head_dim=d`：
  - 满足 `C = n_heads * head_dim`。MiniMind 默认 768 / 8 = 96。
- `num_key_value_heads=n_kv_heads`：
  - 支持 **GQA (Grouped Query Attention)**：当 `n_kv_heads < n_heads` 时，多个 Query 头共享一组 KV 头，显著减少显存占用并提升推理速度。
- **MoE 配置**：
  - `num_experts`：专家总数（默认 4）。
  - `num_experts_per_tok=k`：每个 Token 激活的专家数（默认 1）。
  - `router_aux_loss_coef`：路由均衡损失系数，防止所有 Token 都挤向同一个专家。

---

## 3. `RMSNorm`：稳健的归一化 (L07)

```python
49: class RMSNorm(torch.nn.Module):
...
55:     def norm(self, x):
56:         return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

维度追踪：

- 输入 `x` 为 `[..., dim]`。
- 第 56 行：`x.pow(2).mean(-1, keepdim=True)` -> `[..., 1]`。
- `rsqrt` 计算均方根倒数，与 `x` 广播相乘，输出形状不变。
- **动机**：相比 LayerNorm 减均值再除方差，RMSNorm 只除均方根，计算更简单且在 LLM 中已被证明效果持平。

---

## 4. RoPE 频率预计算：`precompute_freqs_cis`（维度推导）

维度推导（假设 `dim=head_dim=d`）：

1. `freqs` 初始形状 `[d/2]`（只取偶数索引）。
2. `t = [end]` (最大序列长度)。
3. `freqs = outer(t, freqs)` -> `[end, d/2]`。
4. `torch.cat([cos, cos], dim=-1)` -> `[end, d]`。
5. 返回的 `freqs_cos/sin` 形状均为 `[max_seq_len, head_dim]`。

---

## 5. Attention：核心张量维度追踪

这是整个模型最复杂的维度变换区。

### 5.1 QKV 投影与切分

```python
111: xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
112: xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
113: xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
```

- `xq`: `[B, T, n_heads, d]`
- `xk/xv`: `[B, T, n_kv_heads, d]`

### 5.2 QK-Norm (MiniMind 特色)

```python
115: xq, xk = self.q_norm(xq), self.k_norm(xk)
```

- **动机**：在旋转位置编码前对 Q 和 K 分别进行 RMSNorm。这借鉴了 QK-Norm 思想，能有效抑制训练中期的梯度爆炸，提升训练稳定性。

### 5.3 GQA 下的 KV 广播

```python
122: xq, xk, xv = (xq.transpose(1, 2), 
                  repeat_kv(xk, self.n_rep).transpose(1, 2), 
                  repeat_kv(xv, self.n_rep).transpose(1, 2))
```

- `repeat_kv(xk, n_rep)`：
  - `[B, T, n_kv_heads, d] -> [B, T, n_heads, d]`
- `transpose(1, 2)`：
  - -> `[B, n_heads, T, d]`
- **最终对齐**：Q, K, V 全部对齐为 `[B, n_heads, T, d]`，准备进行点积注意力。

---

## 6. MoE 路由逻辑：`MOEFeedForward`（关键行解析）

```python
157: scores = F.softmax(self.gate(x_flat), dim=-1) # [BT, E]
158: topk_weight, topk_idx = torch.topk(scores, k=..., dim=-1) # [BT, k]
166: y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight))
```

- **路由机制**：
  - `self.gate` 将隐藏层映射到专家数量 `E`。
  - `softmax` 得到每个 Token 对各专家的“匹配度”。
  - `index_add_`：这是一个高效的散射操作。它只对选中该专家的 Token 执行 `expert(x)` 计算，并将结果按权重累加回总输出 `y`。
- **辅助损失 (Aux Loss)**：
  - 用于衡量专家选择的分布熵。如果所有 Token 都选同一个专家，`aux_loss` 会变大，从而迫使模型均匀分配专家负载。

---

## 7. 生成逻辑与 KV Cache

```python
256: past_len = past_key_values[0][0].shape[1] if past_key_values else 0
257: outputs = self.forward(input_ids[:, past_len:], past_key_values=past_key_values, ...)
```

- **增量推理 (Incremental Inference)**：
  - 在生成第 `t` 个 Token 时，我们不再重新计算前 `t-1` 个 Token 的 QKV。
  - `input_ids[:, past_len:]`：只取当前步那一个 Token 的 ID 传入 `forward`。
  - `past_key_values`：从 Cache 中读取历史 K 和 V，在 `Attention` 内部进行拼接：
    - `xk = torch.cat([past_k, xk_new], dim=1)` -> `[B, T_total, n_kv_heads, d]`。
  - 这种机制将推理复杂度从 $O(T^2)$ 降低到 $O(T)$。

---

## 8. 训练 Loss 的对齐 (Weight Tying)

```python
234: self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
235: self.model.embed_tokens.weight = self.lm_head.weight
```

- **权重共享**：输入层的 Embedding 矩阵与输出层的 LM Head 投影矩阵共享同一份内存参数。
- **好处**：
  1. 减少了模型总参数量（减少了 `vocab_size * hidden_size` 个参数）。
  2. 理论上可以加快模型收敛，因为词向量和预测头在同一个语义空间更新。

---

## 9. 本章总结

通过本章，你应该能脑补出数据流动的完整“形状图”：
1. `[B, T]` 的 ID 序列进入 Embedding。
2. 变为 `[B, T, C]` 的连续向量。
3. 经由 `MiniMindBlock` 的多层处理，始终保持 `[B, T, C]`。
4. 每一层内部，QKV 经投影、QK-Norm、RoPE 旋转、GQA 扩展后进行注意力计算。
5. 经 `lm_head` 映射为 `[B, T, V]` 的 Logits。
6. 训练时通过错位对齐计算交叉熵 Loss。

下一章 `02_LoRA 适配与合并` 将解析如何在这个闭环的 `[B, T, C]` 路径上横向切开，插入低秩旁路。
