import math, torch, torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind 配置类
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            hidden_size=768,  # 模型隐藏层维度 (Embedding 维度)
            num_hidden_layers=8,  # Transformer 层数
            use_moe=False,  # 是否启用混合专家模型 (MoE)
            **kwargs
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.use_moe = use_moe
        self.dropout = kwargs.get("dropout", 0.0)  # 全局 Dropout 概率
        self.vocab_size = kwargs.get("vocab_size", 6400)  # 词表大小，默认 6400 (轻量级)
        self.bos_token_id = kwargs.get("bos_token_id", 1)  # 句子开始符 ID
        self.eos_token_id = kwargs.get("eos_token_id", 2)  # 句子结束符 ID
        self.flash_attn = kwargs.get("flash_attn", True)  # 是否启用 Flash Attention 优化
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)  # 注意力头数
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)  # KV 头数 (用于 GQA)
        # 每个 Head 的维度 = hidden_size // num_attention_heads (例如 768//8 = 96)
        self.head_dim = kwargs.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.hidden_act = kwargs.get("hidden_act", 'silu')  # 激活函数，默认 SiLU (Swish)
        # MLP 中间层维度，默认计算公式：ceil(h * pi / 64) * 64
        self.intermediate_size = kwargs.get("intermediate_size", math.ceil(hidden_size * math.pi / 64) * 64)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 32768)  # 最大序列长度
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)  # RMSNorm 的 epsilon，防止除零
        self.rope_theta = kwargs.get("rope_theta", 1e6)  # RoPE 的底数 theta
        self.inference_rope_scaling = kwargs.get("inference_rope_scaling", False)  # 推理阶段是否启用 YaRN 插值缩放
        # YaRN (Yet another RoPE extensioN) 配置，用于长文本外推
        self.rope_scaling = {
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 16,
            "original_max_position_embeddings": 2048,
            "attention_factor": 1.0,
            "type": "yarn"
        } if self.inference_rope_scaling else None

        ### MoE 专属配置 (当 use_moe = True 时有效)
        self.num_experts = kwargs.get("num_experts", 4)  # 总专家数
        self.num_experts_per_tok = kwargs.get("num_experts_per_tok", 1)  # 每个 token 激活的专家数 (Top-K)
        self.moe_intermediate_size = kwargs.get("moe_intermediate_size", self.intermediate_size)  # 专家内部维度
        self.norm_topk_prob = kwargs.get("norm_topk_prob", True)  # 是否对 Top-K 概率进行归一化
        self.router_aux_loss_coef = kwargs.get("router_aux_loss_coef", 5e-4)  # 路由均衡辅助损失系数


# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏
#                                     MiniMind 模型实现
# 🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏🌎🌍🌏

class RMSNorm(torch.nn.Module):
    """
    均方根归一化 (Root Mean Square Layer Normalization)
    相比 LayerNorm，去除了均值中心化，仅保留标准差缩放，计算开销更低。
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习的缩放参数 gamma

    def norm(self, x):
        # 计算公式：x * (1 / sqrt(mean(x^2) + eps))
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 输入 x: [..., dim], 输出与输入形状一致
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    """
    预计算旋转位置嵌入 (RoPE) 的正余弦频率矩阵。
    dim: head_dim (每个注意力头的维度)
    end: 最大序列步数
    rope_scaling: 是否应用 YaRN 长文本外推插值
    """
    # 计算频率序列: 1 / (base ^ (2i/dim))
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0

    if rope_scaling is not None:
        # YaRN 逻辑: 通过线性斜坡函数重新缩放频率，以支持训练长度之外的推理
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048),
            rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0),
            rope_scaling.get("beta_slow", 1.0),
            rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            # 计算斜坡函数 ramp，用于平滑过渡频率缩放
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)

    # 构造时间步向量 t: [0, 1, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 计算外积得到频率矩阵: [end, dim/2]
    freqs = torch.outer(t, freqs).float()
    # 拼接得到完整的 cos 和 sin 矩阵: [end, dim]
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    将旋转位置嵌入应用于 Query 和 Key 向量。
    q, k 形状: [bs, seq_len, n_heads, head_dim] 或 [bs, n_heads, seq_len, head_dim]
    cos, sin 形状: [seq_len, head_dim]
    """

    def rotate_half(x):
        # 将向量后半部分取反并与前半部分互换: [-x2, x1]
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    # 旋转公式: x_rotated = x * cos + rotate_half(x) * sin
    # unsqueeze(unsqueeze_dim) 用于匹配 batch 和 heads 的维度
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    针对 Grouped Query Attention (GQA)，将 KV 头复制 n_rep 次以匹配 Q 头数。
    x: [bs, slen, n_kv_heads, head_dim]
    返回: [bs, slen, n_kv_heads * n_rep, head_dim]
    """
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 利用 expand 和 reshape 进行高效维度扩展
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    """
    多头注意力/分组查询注意力 (MHA/GQA) 实现
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # 确定 KV 头数，若未指定则与注意力头数一致
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        # Q 与 KV 的头数比例 (GQA 的倍数)
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim

        # QKV 线性变换层
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        # 输出线性投影层
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)

        # QK-Norm: 对旋转前的 QK 进行归一化，增强训练稳定性 (MiniMind 特色)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        # 检查是否可以使用 PyTorch 原生的 Flash Attention (scaled_dot_product_attention)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # x: [batch_size, seq_len, hidden_size]
        bsz, seq_len, _ = x.shape

        # 1. 投影到 Q, K, V
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        # 2. Reshape 为多头格式: [bs, seq_len, n_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # 3. 应用 QK-Norm
        xq, xk = self.q_norm(xq), self.k_norm(xk)

        # 4. 应用旋转位置嵌入 (RoPE)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        # 5. KV Cache 逻辑: 拼接历史 KV
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 6. 为 GQA 扩展 KV 头并转置维度: [bs, n_heads, seq_len, head_dim]
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        # 7. 计算 Attention Score
        if self.flash and (seq_len > 1) and (past_key_value is None) and (
                attention_mask is None or torch.all(attention_mask == 1)):
            # 使用高效的 Flash Attention
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自动应用因果遮掩 (Causal Mask)
            )
        else:
            # 标准手动 Attention 计算
            # scores: [bs, n_heads, seq_len, full_seq_len]
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            # 应用 Causal Mask: 将右上角置为 -inf
            scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            # 应用外部 Attention Mask (处理 Padding)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            # Softmax + Dropout + 乘 V
            output = self.attn_dropout(F.softmax(scores.float(), dim=-1).type_as(xq)) @ xv

        # 8. 恢复维度并输出: [bs, seq_len, hidden_size]
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    """
    标准 SwiGLU 激活的前馈神经网络 (LLaMA 风格)
    """

    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        # 门控线性单元 (GLU) 的三部分投影
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 计算流程: Down(Act(Gate(x)) * Up(x))
        # 激活通常是 SiLU
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MOEFeedForward(nn.Module):
    """
    混合专家前馈网络 (MoE)
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由层: 输入 token 映射到专家数量的得分
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        # 专家列表: 每一层专家都是一个标准的 FeedForward
        self.experts = nn.ModuleList(
            [FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)  # 展平为 [N_tokens, hidden_dim]

        # 1. 路由计算: 计算每个 token 对专家的选择概率
        scores = F.softmax(self.gate(x_flat), dim=-1)
        # 选取 Top-K 个专家
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            # 重新归一化选中的专家权重，使其和为 1
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        y = torch.zeros_like(x_flat)
        # 2. 专家分发与计算:
        # 为了效率，逐个专家处理选中它的 tokens
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)  # 哪些 token 的 top-k 中包含了专家 i
            if mask.any():
                # 找到被分派到该专家的 token 索引
                token_idx = mask.any(dim=-1).nonzero().flatten()
                # 提取对应权重
                weight = topk_weight[mask].view(-1, 1)
                # 计算专家输出并加权累加回 y
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                # 梯度辅助逻辑，防止未选中的专家在分布式训练中出现参数同步问题
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())

        # 3. 负载均衡损失 (Auxiliary Loss)
        if self.training and self.config.router_aux_loss_coef > 0:
            # 理想状态下专家负载应均匀分布 (load)
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()

        return y.view(batch_size, seq_len, hidden_dim)


class MiniMindBlock(nn.Module):
    """
    单个 Transformer 层 (Decoder Block)
    包含: Attention -> Add -> Norm -> MLP/MoE -> Add
    采用 Pre-Norm 结构
    """

    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 根据配置决定使用标准 MLP 还是 MoE
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 1. Self Attention 路径 (残差连接)
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # Pre-Norm
            position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual

        # 2. MLP/MoE 路径 (残差连接)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    """
    MiniMind 模型主体
    """

    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        # 词嵌入层: [vocab_size, hidden_size]
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        # 堆叠 num_hidden_layers 个 Transformer 层
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        # 输出前的最终归一化层
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 预计算 RoPE 频率矩阵并注册为 buffer (不参与梯度更新)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_position_embeddings,
            rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_length = input_ids.shape

        # 处理 KV Cache 列表
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        # 当前生成的起始位置 (如果是增量推理，start_pos > 0)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        # 1. 获取词向量
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # 2. 提取当前片段对应的 RoPE 嵌入
        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        # 3. 逐层前向传播
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        # 4. 最终归一化
        hidden_states = self.norm(hidden_states)

        # 5. 汇总所有 MoE 层的辅助损失
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)],
                       hidden_states.new_zeros(1).squeeze())

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    """
    MiniMind 因果语言模型封装，用于训练和生成
    兼容 HuggingFace 的 PreTrainedModel 和 GenerationMixin
    """
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        # 语言模型头 (预测下一个 token 的概率分布)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享 (Weight Tying): Embedding 层与 LM Head 共享参数，显著减少模型参数量
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None,
                **kwargs):
        """
        前向传播计算 Logits 和 Loss
        """
        # 1. 运行主体模型
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache,
                                                             **kwargs)

        # 2. 计算 Logits
        # logits_to_keep: 通常只保留最后一个 token 的 logits 用于生成以节省显存
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        # 3. 计算交叉熵损失 (如果提供了 labels)
        if labels is not None:
            # 错位对齐: logits[t] 预测 labels[t+1]
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)

        # 返回符合 HF 格式的输出对象
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states
        )

    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50,
                 eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True,
                 repetition_penalty=1.0, **kwargs):
        """
        推理生成逻辑 (支持 KV Cache 和多种采样策略)
        """
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        # 标记哪些 sequence 已经生成结束
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        if streamer: streamer.put(input_ids.cpu())

        # 自回归生成循环
        for _ in range(max_new_tokens):
            # 仅处理最新生成的 token，结合 KV Cache
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache,
                                   **kwargs)

            # 更新 Attention Mask 以匹配新的序列长度
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)],
                                       -1) if attention_mask is not None else None

            # 获取最后一个 token 的 Logits 并应用 Temperature 缩放
            logits = outputs.logits[:, -1, :] / temperature

            # 应用重复惩罚 (Repetition Penalty)
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    logits[i, torch.unique(input_ids[i])] /= repetition_penalty

            # Top-K 采样过滤
            if top_k > 0:
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')

            # Top-P (核采样) 过滤
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')

            # 采样或取最大概率的 token
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(
                logits, dim=-1, keepdim=True)

            # 如果序列已结束，则填充 EOS
            if eos_token_id is not None:
                next_token = torch.where(finished.unsqueeze(-1),
                                         next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)

            # 拼接新 token 并更新 KV Cache
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None

            if streamer: streamer.put(next_token.cpu())

            # 检查是否所有序列都已生成 EOS
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break

        if streamer: streamer.end()
        # 若需要返回 KV Cache (用于后续对话上下文衔接)
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
