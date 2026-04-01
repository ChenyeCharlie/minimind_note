import os
import sys

# 设置当前包名为 trainer，并添加上级目录路径以支持导入 model 和 dataset
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import math
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

# 忽略警告信息
warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    重复惩罚项：检测生成的文本中 n-gram 的重复比例。
    用于在强化学习中抑制模型产生复读机现象。
    """
    # 简单的正则分词
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    # 构造 n-gram 序列
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    # 计算重复度并截断到 cap
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


class CriticModel(MiniMindForCausalLM):
    """
    自定义 Critic 模型：
    继承自 MiniMindForCausalLM，但将最后的输出头 (lm_head) 替换为输出标量价值的线性层 (value_head)。
    用于估计当前状态 (Prompt + 已生成 Token) 的期望回报。
    """
    def __init__(self, params):
        super().__init__(params)
        # 价值头：将隐藏状态映射到 1 维标量
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向传播计算价值估计 V(s)。
        Output: [batch_size, seq_len] - 每个 token 位置的价值估计
        """
        # 1. 调用基类的 Transformer 提取特征
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 2. 归一化隐藏状态
        hidden_states = self.model.norm(outputs[0])
        # 3. 线性映射并去掉最后一维，得到形状为 [B, S] 的 Value
        values = self.value_head(hidden_states).squeeze(-1)
        return values


def calculate_rewards(prompts, responses, reward_model):
    """
    计算外部奖励分：
    结合了奖励模型 (Reward Model) 的判别分、长度惩罚、重复惩罚以及思考链逻辑奖惩。
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 1. 解析 Prompt 还原对话历史消息格式
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            
            answer = response
            # 2. 长度奖惩：鼓励生成适中长度的回复
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
            
            # 3. 思考链 (Reasoning) 逻辑奖惩
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                # 鼓励有长度的思考
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                # 惩罚格式错误（如出现多个 </think> 标签）
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()
            
            # 4. 重复惩罚：扣除重复性得分
            rewards[i] -= rep_penalty(answer)

            # 5. 调用外部奖励模型 (判别模型) 获取质量分数
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        # 最终奖励 = 规则项 + 奖励模型分
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):
    """
    PPO 训练主循环。
    执行步骤：采样 (Rollout) -> 计算奖励与优势 (GAE) -> 策略与价值迭代。
    """
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0

    for step, batch in enumerate(loader, start=start_step + 1):
        # ========== 1. 在线采样阶段 (Rollout) ==========
        prompts = batch["prompt"]
        # 使用左侧 Padding 编码 Prompt，便于生成
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)
        prompt_length = enc.input_ids.shape[1]

        # 调用引擎采样回复
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids # 形状 [B, P+R]
        responses_text = rollout_result.completions
        
        # 计算外部奖励分数 [B]
        rewards = calculate_rewards(prompts, responses_text, reward_model)

        # DEBUG 打印
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] Reward: {rewards[i].item():.4f}")

        # ========== 2. 优势估计与数据对齐 (GAE) ==========
        full_mask = (gen_out != tokenizer.pad_token_id).long()
        # 训练目标对齐 (Shift): gen_out[t] 预测 gen_out[t+1]
        labels = gen_out[:, 1:].clone()
        seq_len, resp_start = gen_out.size(1) - 1, prompt_length - 1
        
        # 构造掩码，只在回复 (Response) 区间且非 Padding 位置计算梯度
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= resp_start
        final_mask = (resp_mask & (~labels.eq(tokenizer.pad_token_id))).float()
        
        B = len(prompts)
        # 精确计算每个 Response 的有效长度（处理 EOS）
        resp_labels = labels[:, resp_start:]
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)
        resp_pad_mask = ~resp_labels.eq(tokenizer.pad_token_id)
        resp_lengths = resp_pad_mask.sum(dim=1)
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask
        has_eos = eos_mask.any(dim=1)
        eos_pos = torch.argmax(eos_mask.int(), dim=1)
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()
        resp_value_mask = resp_policy_mask.clone()

        with torch.no_grad():
            # 2.1) 获取旧策略下的 Logprob 和 Value
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
            old_resp_values = values_seq[:, resp_start:-1] * resp_value_mask # [B, R]
            
            actor_for_rollout = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            with autocast_ctx:
                logits = actor_for_rollout(input_ids=gen_out, attention_mask=full_mask).logits
            
            # 计算生成 token 的对数概率
            old_resp_logp = F.log_softmax(logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)[:, resp_start:]
            
            # 2.2) 获取参考模型的 Logprob 用于计算 KL 惩罚 (隐式包含在 PPO Loss 中或显式扣除)
            ref_logp_all = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
            ref_resp_logp = ref_logp_all[:, resp_start:]
            
            # 2.3) 计算 GAE (Generalized Advantage Estimation)
            token_rewards = torch.zeros_like(old_resp_logp)
            last_idx = resp_lengths - 1
            # 将外部奖励加在回复的最后一个非 Pad token 位置
            token_rewards[torch.arange(B, device=args.device), last_idx] += rewards

            gen_len = old_resp_values.size(1)
            lastgaelam = torch.zeros(B, device=args.device)
            advs_rev = []
            # 逆向遍历序列计算 TD-Error 和优势值
            for t in reversed(range(gen_len)):
                next_v = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                # TD误差: δ = r + γ * V(s_{t+1}) - V(s_t)
                delta = token_rewards[:, t] + args.gamma * next_v - old_resp_values[:, t]
                # GAE: A = δ + (γ * λ) * A_{next}
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            
            advantages = torch.stack(advs_rev[::-1], dim=1) # [B, R]
            # 目标收益 (Target Value): Returns = Advantage + Baseline_Value
            returns = advantages + old_resp_values # [B, R]

            # 归一化优势值，加速训练收敛
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # ========== 3. PPO 更新阶段 (Epochs & Minibatches) ==========
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False
        log_metrics = {"p_loss": 0, "v_loss": 0, "kl": 0, "kl_ref": 0, "log_count": 0}
        
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
        
        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo: break
            b_inds = torch.randperm(B, device=args.device)
            
            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]
                
                # 3.1) 计算当前策略的 Value
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq[:, resp_start:-1]

                # 3.2) 计算当前策略的 Logprob
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

                mb_logp_all = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1)
                mb_resp_logp = mb_logp_all[:, resp_start:]
                
                # 3.3) 计算 KL 散度与早停判断
                log_ratio = mb_resp_logp - old_resp_logp[inds]
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                # 分布式下同步 KL 状态，防止某些 rank 先退出导致死锁
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                
                # 3.4) 计算 PPO Policy Loss (Clipped Surrogate Objective)
                ratio = torch.exp(log_ratio)
                # 概率比例裁剪
                surr1 = -advantages[inds] * ratio
                surr2 = -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon)
                
                # 计算与参考模型的 KL 惩罚，防止模型退化
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                policy_loss = (torch.max(surr1, surr2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                policy_loss = policy_loss + args.kl_coef * kl_ref_penalty
                
                # 3.5) 计算 Value Loss (MSE with Clipping)
                v_err = mb_resp_values - returns[inds]
                v_err_clipped = torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value, old_resp_values[inds] + args.cliprange_value) - returns[inds]
                value_loss = 0.5 * (torch.max(v_err**2, v_err_clipped**2) * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                # 总 Loss = 策略 Loss + 系数 * 价值 Loss + MoE 辅助 Loss
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0 # 保持梯度图闭环
                else:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()

                # 指标累计
                log_metrics["p_loss"] += policy_loss.item()
                log_metrics["v_loss"] += value_loss.item()
                log_metrics["kl"] += approx_kl_val.item()
                log_metrics["log_count"] += 1
                grad_accum_step += 1

                # 3.6) 梯度更新
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # 处理末尾梯度
        if grad_accum_step % args.accumulation_steps != 0:
            actor_optimizer.step(); critic_optimizer.step()
            actor_scheduler.step(); critic_scheduler.step()
            actor_optimizer.zero_grad(); critic_optimizer.zero_grad()
        
        # 周期性同步最新的策略模型给 Rollout 引擎
        if is_main_process() and step % args.save_interval == 0: 
            rollout_engine.update_policy(actor_model)

        # ========== 4. 日志记录与模型保存 ==========
        if is_main_process():
            reward_avg = rewards.mean().item()
            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), Reward: {reward_avg:.4f}, Actor LR: {actor_optimizer.param_groups[0]['lr']:.8f}")
            if wandb: wandb.log({"reward": reward_avg, "kl": log_metrics["kl"]/max(log_metrics["log_count"], 1)})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存 Actor 权重和完整检查点
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)

        # 显存清理
        del enc, gen_out, rollout_result, rewards, advantages, returns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否加速编译")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下打印间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8997", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置与 Checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 日志系统 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 模型定义 (Actor-Critic 架构) ==========
    base_weight = args.from_weight
    # Actor：执行策略 (Policy) 的 LLM
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference：计算 KL 散度的基准 LLM (冻结)
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    
    # Critic：价值估计模型 (初始化为 Actor 的权重，但输出头不同)
    critic_model = CriticModel(lm_config)
    # 加载预训练权重到 Critic (忽略不匹配的头)
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)
    
    # 奖励模型：评估回复好坏
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    
    # ========== 6. 引擎、数据与调度器 ==========
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总迭代步数用于 Cosine 调度
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)
    
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # 从断点恢复
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 优化与分布式 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    
    if is_main_process(): 
        rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始主循环 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        
        ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理 ==========
    if dist.is_initialized(): dist.destroy_process_group()
