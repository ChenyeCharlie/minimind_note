"""
Rollout Engine - 强化学习采样与推理引擎
本模块实现了可插拔的推理引擎，支持 PyTorch 原生推理以及通过 SGLang (高性能推理框架) 进行采样。
"""
import os
import sys

# 设置包名并添加搜索路径
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    计算序列中最后 n_keep 个 token 的对数概率 (log-probabilities)。
    在 RL (如 PPO/GRPO) 中，需要计算模型生成部分的 logprob 以进行优势估计或 KL 散度计算。
    """
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    # 1. 解包 DDP 模型，获取底层模型引用
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    
    # 2. 推理模式兼容性处理：如果 input_ids 处于推理张量状态，则克隆一份以允许梯度或进一步操作
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids
    
    # 3. 前向传播：
    # - logits_to_keep=n_keep + 1：为了计算 n 个 token 的概率，我们需要前一个位置的 logits。
    # - [:, :-1, :]：切片去掉最后一个 token 的预测 logits，因为它没有对应的目标 token id。
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    
    per_token_logps = []
    # 4. 遍历 batch 中的每一行，根据对应的 token id 提取 logprob
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        # - log_softmax：将 logits 转换为对数概率
        # - torch.gather：根据实际出现的 ids_row，从词表维 (dim=-1) 提取对应的概率值
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    # 5. 返回形状为 [batch, n_keep] 的张量
    return torch.stack(per_token_logps)


@dataclass
class RolloutResult:
    """
    Rollout (采样) 结果的数据类。
    包含：完整序列 ID、生成的回复 ID、每个生成的 token 的 logprob、以及解码后的文本。
    """
    output_ids: Tensor      # 完整 ID [B, Prompt_Len + Completion_Len]
    completion_ids: Tensor  # 仅回复 ID [B, Completion_Len]
    per_token_logps: Tensor # 回复部分的 logprob [B, Completion_Len]
    completions: List[str]  # 解码后的字符串列表


class RolloutEngine(ABC):
    """推理引擎抽象基类，定义了统一的采样和模型更新接口。"""
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """执行采样操作。"""
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """同步最新的模型参数到推理引擎。"""
        pass


class TorchRolloutEngine(RolloutEngine):
    """
    基于 PyTorch 原生 generate 方法的推理引擎。
    适用于单机、小模型或不需要极致吞吐量的场景。
    """
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx # 混合精度上下文
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 获取基础模型
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        
        # 1. 生成阶段：关闭梯度以节省显存
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                num_return_sequences=num_generations, # 每个 prompt 生成多少个回复
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 2. 提取回复部分
        prompt_len = prompt_ids.size(1)
        completion_ids = output_ids[:, prompt_len:]
        
        # 3. 计算 Log-probabilities：需要在对应的精度上下文(autocast)下计算
        from contextlib import nullcontext
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()
        with ctx:
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1))
        
        # 4. 解码文本并封装结果
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        return RolloutResult(output_ids, completion_ids, per_token_logps, completions)
    
    def update_policy(self, model: torch.nn.Module):
        """PyTorch 引擎直接更新引用即可。"""
        self.policy_model = model


class SGLangRolloutEngine(RolloutEngine):
    """
    基于 SGLang HTTP 服务的高性能推理引擎。
    通过并行推理大幅提升强化学习中的采样速度。
    """
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path # 模型权重中转路径
        self.timeout = timeout
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        # 1. 数据清理：去除左侧 padding tokens，SGLang 接口通常接收有效 id 列表
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()
            input_ids_list.append(valid_ids)
        # 扩展列表以匹配生成倍数
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        # 2. 构造请求 Payload
        payload = {
            "input_ids": all_input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True, # 要求返回每个 token 的 logprob
        }
        
        # 3. 发送请求并解析结果
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        
        results = resp.json()
        if not isinstance(results, list): results = [results]
        
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            meta = result.get("meta_info", {})
            # 提取生成的 id 和 logprobs
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            # 解析 SGLang 返回的 logprob 格式 (可能是 [prob, id] 的列表)
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])
                elif isinstance(item, (int, float)):
                    logprobs.append(item)
            
            prompt = all_input_ids[i]
            all_output_ids.append(prompt + completion_ids)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        # 4. 后处理：将不同长度的列表 Pad 成对齐的 Tensor
        device = prompt_ids.device
        max_out_len = max(len(ids) for ids in all_output_ids)
        max_comp_len = max(len(ids) for ids in all_completion_ids)
        max_logp_len = max(len(lp) for lp in all_logprobs)
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len),
            per_token_logps=pad_to_tensor(all_logprobs, max_logp_len, pad_val=0.0),
            completions=completions,
        )
    
    def update_policy(self, model: torch.nn.Module):
        """
        SGLang 模型同步逻辑：
        1. 将当前训练中的模型权重保存到本地磁盘。
        2. 调用 SGLang API 的 update_weights_from_disk 接口让推理服务热加载新权重。
        """
        unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        abs_path = os.path.abspath(self.shared_ckpt_path)
        # 克隆参数以防在保存过程中主模型被意外修改
        unwrapped.lm_head.weight = torch.nn.Parameter(unwrapped.lm_head.weight.clone())
        state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
        # 物理保存到磁盘中转站
        unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
        unwrapped.model.embed_tokens.weight = unwrapped.lm_head.weight
        self.tokenizer.save_pretrained(abs_path)
        
        # 通知 SGLang 刷新权重
        resp = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": abs_path},
            timeout=self.timeout
        )
        if resp.status_code != 200: 
            print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
        return resp.status_code == 200
    
    def flush_cache(self) -> bool:
        """清除 SGLang 服务端的推理缓存。"""
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        """检查 SGLang 推理服务是否健康。"""
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """引擎工厂函数：根据配置返回对应的推理引擎实例。"""
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
