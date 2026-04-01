import os
import sys

# 设置当前包名为 trainer，并添加上级目录路径以支持导入 model 和 dataset
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import gc
import json
import math
import random
import signal
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# ================================ 工具与 Reward 相关逻辑 = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    """
    重复惩罚项：计算文本中 n-gram 的重复比例。
    用于在强化学习中惩罚产生复读机现象的模型。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义：包含函数名、描述及参数模式 (JSON Schema) ========
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 工具执行的模拟数据 (Mock Data) ========
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行逻辑：根据函数名调用对应的模拟计算 ========
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验逻辑：检查模型生成的参数是否满足基本要求 ========
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行辅助函数 ========
def parse_tool_calls(text):
    """从生成的文本中提取 <tool_call> 标签内的 JSON 内容。"""
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: 
            calls.append(json.loads(m.strip()))
        except: 
            pass
    return calls

def execute_tool(name, args):
    """
    根据函数名执行对应的 Mock 函数。
    包含 signal 定时器以防止 eval 等操作导致的死循环或超时。
    """
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        # 设置 1 秒超时
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass

# ======== 多轮自回归采样逻辑 (Sequential Rollout) ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """
    针对 Agent 场景的单样本多轮采样逻辑。
    1. 模型生成 Assistant 内容。
    2. 若包含工具调用，解析并模拟执行，将结果追加到 Context 中。
    3. 重复直至无工具调用或达到最大轮数。
    """
    all_outputs = [] # 记录每一轮生成的文本
    prompt_ids = None # 初始 Prompt 的 ID 序列
    response_ids = [] # 存储模型生成的 ID (用于 RL 更新)
    response_mask = [] # 标记哪些 ID 是模型生成的 (参与 Loss)，哪些是环境反馈的 (Mask=0)
    response_old_logps = [] # 存储旧策略下的对数概率
    final_context = ""
    unfinished = False
    open_thinking = random.random() < thinking_ratio
    
    for turn in range(max_turns):
        # 1. 构造当前轮次的 Prompt
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        
        if prompt_ids is None:
            prompt_ids = inputs["input_ids"][0].tolist()
        
        # 2. 调用推理引擎生成回复
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        
        # 3. 提取生成的 Token ID、Logprobs 和文本
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()
        new_text = rollout_result.completions[0]
        
        # 过滤掉 Padding 和 EOS
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        clean_ids = [t for t, _ in pairs]
        clean_logps = [lp for _, lp in pairs]
        
        all_outputs.append(new_text)
        response_ids.extend(clean_ids)
        response_mask.extend([1] * len(clean_ids)) # 模型生成部分 Mask = 1
        response_old_logps.extend(clean_logps)
        
        final_context = context + new_text
        
        # 4. 解析工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            # 如果没有工具调用，对话正常结束
            break
            
        unfinished = turn == max_turns - 1
        # 将 Assistant 的生成内容加入消息历史
        messages.append({"role": "assistant", "content": new_text})
        
        # 5. 执行工具并获取反馈
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            # 格式化工具返回结果，限制长度防止爆显存
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
            messages.append({"role": "tool", "content": result_str})

        # 6. 环境反馈同步：获取包含工具返回后的完整序列，并提取增量 ID
        # 增量 ID 是“环境反馈”的内容，其 Mask 设为 0，不计算 Actor Loss
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:] # 提取工具返回部分的 Token ID
        
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta)) # 工具返回部分 Mask = 0
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    # 返回最终结果，包含完整的序列构造元数据
    return all_outputs[-1] if all_outputs else "", final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """批量执行单样本的多轮 Rollout。"""
    all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished = [], [], [], [], [], [], [], []
    
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            msgs_copy = [dict(m) for m in messages]
            c, ctx, p_ids, r_ids, r_mask, r_lp, t_out, unf = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(c); all_contexts.append(ctx); all_prompt_ids.append(p_ids); all_response_ids.append(r_ids); all_response_masks.append(r_mask); all_response_old_logps.append(r_lp); all_turn_outputs.append(t_out); all_unfinished.append(unf)
            
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== 奖励计算 (Reward Logic) ========
def validate_gt_in_text(text, gt_list):
    """
    验证 Ground Truth (gt) 是否存在于生成的文本中。
    支持字符串匹配和数值近似匹配（处理逗号和精度）。
    """
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """
    Agent 强化学习的奖励函数：
    1. 标签闭合分：检查 <tool_call> 等标签是否配对。
    2. 无工具调用路径：执行长度、格式、思考链及 RM (奖励模型) 综合打分。
    3. 有工具调用路径：重点检查工具调用参数的有效性、调用的数量是否正确、以及最终答案是否命中 GT。
    """
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False
        
        # 提取回复中非思考链的部分
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()
        
        valid_names = {t['function']['name'] for t in tools} if tools else set()
        tool_calls = []
        for turn_answer in turn_answers: 
            tool_calls.extend(parse_tool_calls(turn_answer))
            
        # 扣除标签不闭合的分数
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)
        
        # -------- 分支 1：模型没有进行工具调用 --------
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5
            if '</think>' in response:
                think, _ = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5
                reward += 0.25 if response.count('</think>') == 1 else -0.25
            
            # 使用 RM 模型获取基础能力奖励分
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                msg_history = [{"role": role, "content": content.strip()} for role, content in matches]
                reward += reward_model.get_score(msg_history, answer)
            
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)
            
        # -------- 分支 2：模型执行了工具调用 --------
        else:
            gt = gt_batch[sample_idx]
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                check = CHECK_ARGS.get(name)
                # 统计有效的工具调用 (名称匹配、参数合法)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))
            
            # 计算工具调用数与预期 (GT 长度) 的差距
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap
            
            # 提取包含最终结论的文本进行 GT 比对
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            
            if gt: 
                # 根据命中准确值的比例给予高额奖励
                reward += 2.5 * len(verified) / len(gt)
            
            if unfinished: reward -= 0.5
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)
            
    return rewards

# ================================ 训练主逻辑 ================================

def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    """
    Agent RL 训练循环：基于 GRPO 框架。
    """
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']
        tools_batch = batch['tools']
        gt_batch = batch['gt']
        last_step = step

        # 1. 批量多轮采样：获取生成的 Token、Mask 和旧策略 Logprob
        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # 2. 样本对齐与打包：将不同轮次产生的交互拼接成完整的训练序列
        prompts_text = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r
            # Mask 只有模型生成的部分为 1，环境反馈部分为 0
            mask = [0] * len(p) + m
            # Logprob 对齐序列长度 (Shift 后长度减 1)
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
            
            # 超长截断
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))
            
        # 3. 构造批次 Tensor
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)

        # 4. 计算当前策略 Logprobs
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 取 shift 后的 logits 和 input_ids 对应位置的概率
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        # 5. 计算参考策略 Logprobs
        with torch.no_grad():
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1)

        # 6. 处理 EOS 后的掩码：生成结束后不计算 Loss
        completion_mask = full_response_masks[:, 1:]
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        
        # 7. 计算 GRPO 相对优势
        rewards = calculate_rewards(prompts_text, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)
        grouped_rewards = rewards.view(-1, args.num_generations)
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # 8. GRPO 损失计算 (带 KL 散度约束)
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        
        # 概率比率裁剪 Loss
        clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
        per_token_loss = -(torch.min(ratio * advantages.unsqueeze(1), clipped_ratio * advantages.unsqueeze(1)) - args.beta * per_token_kl)
        
        # 应用 Completion Mask 并计算均值
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # 9. 参数更新
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            if is_main_process() and step % args.save_interval == 0: 
                rollout_engine.update_policy(model)

        # 10. 日志与保存
        if step % args.log_interval == 0 or step == iters:
            ar = rewards.mean().item()
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, Loss:{loss.item():.4f}, LR:{optimizer.param_groups[0]["lr"]:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward": ar, "policy_loss": policy_loss.item()})

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            # 保存 Actor 权重和 Resume 检查点
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)

        # 清理显存
        del per_token_logps, ref_per_token_logps, completions, rewards, advantages, completion_mask

    # 处理最后一波梯度更新
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        optimizer.step(); scheduler.step(); optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据精度")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="Prompt截断长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单轮生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="多轮打包后总长度上界")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="Agent 训练集路径")
    parser.add_argument("--num_generations", type=int, default=4, help="组相对奖励生成数 (GRPO 组大小)")
    parser.add_argument("--beta", type=float, default=0.1, help="KL 惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="Loss 计算类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO 裁剪范围")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="上界裁剪 (CISPO)")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基准权重前缀")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否续训")
    parser.add_argument("--use_wandb", action="store_true", help="是否开启监控记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="编译加速")
    parser.add_argument("--debug_mode", action="store_true", help="开启采样内容打印")
    parser.add_argument("--debug_interval", type=int, default=20, help="Debug 打印频率")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="开启思考链的概率")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="RM 路径")
    parser.add_argument("--rollout_engine", type=str, default="sglang", choices=["torch", "sglang"], help="引擎选择")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang Tokenizer 路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang 权重同步路径")
    args = parser.parse_args()

    # ========== 1. 初始化 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 模型与 Checkpoint 配置 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 3. 日志监控初始化 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    # ========== 4. 实例化多模型：Policy, Reference, Reward ==========
    # 策略模型 (可训练)
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # 参考模型 (冻结)
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # 外部奖励模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')
    
    # ========== 5. 引擎、数据加载器与优化器 ==========
    # 创建强化学习采样引擎
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # 自定义 Collate 函数：处理字典格式的 batch 数据
    def collate_fn(batch): 
        return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # 从断点恢复状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 6. DDP 并行与编译配置 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        model = DistributedDataParallel(model, device_ids=[local_rank])
    if is_main_process(): 
        rollout_engine.update_policy(model)

    # ========== 7. 主循环迭代 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler: train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        
        # 执行带有多轮交互逻辑的 Agent 训练
        rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))

    # ========== 8. 清理资源 ==========
    if dist.is_initialized(): 
        dist.destroy_process_group()
