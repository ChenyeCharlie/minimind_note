import torch
from torch import optim, nn



# LoRA (Low-Rank Adaptation) 实现
# 分布式训练中，通常只对特定层应用 LoRA 以减少可训练参数。
# 其核心思想是将权重更新 ΔW 分解为两个低秩矩阵的乘积：ΔW = B * A
# 其中 A 是 [in_features, rank]，B 是 [rank, out_features]


class LoRA(nn.Module):
    """
    定义 LoRA 旁路网络结构。
    输入 x 通过 A 降维到 rank，再通过 B 升维回 out_features。
    """

    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA 的秩 (rank)，通常选 8, 16, 32 等，远小于 hidden_size
        # 矩阵 A: [in_features, rank]
        self.A = nn.Linear(in_features, rank, bias=False)
        # 矩阵 B: [rank, out_features]
        self.B = nn.Linear(rank, out_features, bias=False)

        # 1. 矩阵 A 采用高斯初始化 (std=0.02)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 2. 矩阵 B 采用全 0 初始化。
        # 关键点：B 初始化为 0 保证了在训练开始时，LoRA 旁路 ΔW = B * A = 0，
        # 这样模型初始状态与原预训练模型完全一致。
        self.B.weight.data.zero_()

    def forward(self, x):
        # 计算路径: x -> A -> B
        # Tensor 变化: [bs, seq_len, in_dim] -> [bs, seq_len, rank] -> [bs, seq_len, out_dim]
        return self.B(self.A(x))


def apply_lora(model, rank=16):
    """
    动态地将 LoRA 旁路注入到主模型中。
    MiniMind 默认对所有输入输出维度相等的线性层 (通常是 Attention 投影和部分 MLP) 应用 LoRA。
    """
    for name, module in model.named_modules():
        # 筛选条件：线性层且输入输出维度相等 (例如 hidden_size -> hidden_size)
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 1. 创建 LoRA 实例并移动到与模型相同的设备
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 2. 将 lora 实例挂载到原 linear 模块下
            setattr(module, "lora", lora)

            # 3. 拦截原 linear 的 forward 方法
            original_forward = module.forward

            # 4. 定义新的 forward 逻辑: y = Wx + ΔWx = Wx + BAx
            # 使用显式参数绑定 (Default Argument) 避免闭包捕获变量名导致的 bug
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)

            # 5. 替换原模块的 forward
            module.forward = forward_with_lora


def load_lora(model, path):
    """
    加载保存的 LoRA 权重。
    """
    # 1. 加载权重字典到模型对应的设备
    state_dict = torch.load(path, map_location=model.device)
    # 2. 处理可能存在的 DDP/FSDP 前缀 (module.)
    state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

    # 3. 遍历模型中所有包含 lora 属性的模块并加载参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            # 筛选出属于该模块的 lora 参数并去除前缀名
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            module.lora.load_state_dict(lora_state)


def save_lora(model, path):
    """
    仅提取模型中的 LoRA 参数并保存，不保存主模型权重。
    """
    # 获取原始模型 (处理 torch.compile 后的封装)
    raw_model = getattr(model, '_orig_mod', model)
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 处理 DDP 前缀
            clean_name = name[7:] if name.startswith("module.") else name
            # 提取 LoRA 的 A 和 B 矩阵，转为半精度以节省空间
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    # 保存权重字典
    torch.save(state_dict, path)


def merge_lora(model, lora_path, save_path):
    """
    将 LoRA 权重合并回主模型权重中，生成一个新的全量模型文件。
    数学原理：W_merged = W_original + B * A
    合并后的模型可以直接加载，无需 LoRA 逻辑。
    """
    # 1. 先将 LoRA 权重加载到主模型旁路中
    load_lora(model, lora_path)
    raw_model = getattr(model, '_orig_mod', model)

    # 2. 准备新的 state_dict，先复制非 LoRA 的原始权重 (如 Embedding, Norm, LM_Head)
    state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}

    # 3. 遍历所有模块进行矩阵合并
    for name, module in raw_model.named_modules():
        # 寻找原本挂载了 LoRA 的 Linear 层
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            # 获取原始权重 W
            state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
            if hasattr(module, 'lora'):
                # 核心合并公式: W = W + B @ A
                # 注意：module.lora.B.weight 形状是 [out_dim, rank]，A.weight 是 [rank, in_dim]
                # 矩阵乘法后得到 [out_dim, in_dim]，与原 W 形状一致
                delta_w = (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
                state_dict[f'{name}.weight'] += delta_w

    # 4. 保存合并后的全量模型
    torch.save(state_dict, save_path)
