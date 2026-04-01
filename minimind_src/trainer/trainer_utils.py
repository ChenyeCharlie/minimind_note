"""
训练工具函数集合：包含模型参数统计、分布式初始化、学习率调度、断点保存与加载等功能。
"""
import os
import sys

# 设置包名并添加搜索路径，确保可以导入 model 模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    """
    统计模型参数量，并区分总参数量与单次推理激活的参数量 (针对 MoE 模型)。
    """
    # 总参数量 (M)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    
    # 提取 MoE 相关配置
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    
    # 计算单个专家的参数量 (通过查找特定的专家权重名称)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    
    # 非专家层的参数量
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    
    # 每次推理激活的参数量 = 基础层 + 激活的路由专家 + 共享专家
    active = base + (expert * n_active) + (shared_expert * n_shared)
    
    if active < total: 
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: 
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """判断是否为主进程，用于限制日志输出和文件保存。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """仅在主进程中打印日志内容。"""
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    余弦退火学习率调度逻辑：
    lr_t = lr * (0.1 + 0.45 * (1 + cos(pi * current_step / total_steps)))
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化 DDP (Distributed Data Parallel) 分布式训练环境。
    返回 local_rank (当前进程在节点内的设备索引)。
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非 DDP 模式，直接返回设备 0

    # 初始化进程组，使用 NCCL 后端 (NVIDIA GPU 专用)
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    # 将当前进程绑定到对应的 GPU
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """设置所有随机种子，确保实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 禁用 CUDNN 的非确定性优化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    模型断点保存与恢复函数：
    - 若提供 model，则执行保存逻辑。
    - 若未提供 model，则执行从 checkpoint 加载断点数据的逻辑。
    """
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if lm_config.use_moe else ''
    # 纯权重保存路径
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 训练状态 (optimizer/epoch/step) 恢复路径
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # 保存模式
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        
        # 提取 state_dict 并转换为半精度 CPU 存储以节省磁盘和显存
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}
        
        # 使用原子替换方式保存文件，防止程序崩溃导致文件损坏
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)
        
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构造用于恢复训练的完整元数据
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }
        
        # 处理额外的 state_dict 对象 (如学习率调度器或奖励模型)
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:  
        # 加载模式
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            # 如果训练时的 GPU 数量与当前不符，按比例调整已跑过的 step 数
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    """
    模型初始化入口：
    1. 加载 Tokenizer。
    2. 实例化 MiniMindForCausalLM。
    3. 根据 from_weight 加载预训练权重。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        # 加载权重，strict=False 允许加载部分权重 (例如从预训练加载到微调模型)
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    可跳过指定 Batch 的采样器。
    用于断点恢复训练时，跳过该 Epoch 已经跑过的 Batch。
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        # 处理最后一个不完整的 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """
    奖励模型包装器：
    加载预训练好的判别式模型，用于评估生成回复的质量得分。
    """
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """
        根据对话历史和回复，计算奖励分。
        """
        # 拼接对话历史作为 Context
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query
        
        # 构造评估格式
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        
        # 调用模型底层的 get_score 方法并截断到 [-3, 3] 范围
        score = self.model.get_score(self.tokenizer, eval_messages)
        return max(min(score, 3.0), -3.0)
