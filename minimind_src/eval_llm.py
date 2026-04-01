import time
import argparse
import random
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import *
from trainer.trainer_utils import setup_seed, get_model_params

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

def init_model(args):
    """
    模型初始化函数：支持加载原生 PyTorch 权重或 HuggingFace 格式权重。
    同时也处理 LoRA 权重的挂载。
    """
    # 1. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    # 2. 判断加载来源
    if 'model' in args.load_from:
        # 情况 A: 加载原生训练产生的 .pth 权重
        # 首先构造模型配置对象
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling # 控制是否启用 YaRN 外推
        )
        model = MiniMindForCausalLM(config)
        
        # 拼接权重路径
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # 加载 state_dict 到模型
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
        
        # 3. 如果指定了 LoRA 权重，则动态注入并加载
        if args.lora_weight != 'None':
            # 先将模型转换为包含 LoRA 支路的结构
            apply_lora(model)
            # 加载 LoRA 专用权重
            lora_ckp = f'./{args.save_dir}/{args.lora_weight}_{args.hidden_size}.pth'
            load_lora(model, lora_ckp)
    else:
        # 情况 B: 加载标准的 Transformers 格式模型 (通常用于测试导出的模型)
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    
    # 统计并打印模型参数信息
    get_model_params(model, model.config)
    
    # 4. 转换精度并设置为评估模式，移动到目标设备
    return model.half().eval().to(args.device), tokenizer

def main():
    parser = argparse.ArgumentParser(description="MiniMind 模型推理与评估对话脚本")
    
    # ======== 权重加载参数 ========
    parser.add_argument('--load_from', default='model', type=str, help="模型加载路径 (model=原生pth权重)")
    parser.add_argument('--save_dir', default='out', type=str, help="权重存放目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="pth权重命名前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重命名前缀")
    
    # ======== 模型结构参数 (需与训练时一致) ========
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否为MoE架构")
    
    # ======== 推理超参数 ========
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="是否启用RoPE外推")
    parser.add_argument('--max_new_tokens', default=8192, type=int, help="单次生成最大Token数")
    parser.add_argument('--temperature', default=0.85, type=float, help="采样温度 (越大越随机)")
    parser.add_argument('--top_p', default=0.95, type=float, help="核采样阈值")
    parser.add_argument('--open_thinking', default=0, type=int, help="是否强制开启思考链格式")
    parser.add_argument('--historys', default=0, type=int, help="历史对话携带轮数 (需为偶数)")
    parser.add_argument('--show_speed', default=1, type=int, help="是否显示推理速度 (tokens/s)")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="推理设备")
    
    args = parser.parse_args()
    
    # 预设的自动化测试 Prompts
    test_prompts = [
        '你有什么特长？',
        '为什么天空是蓝色的',
        '请用Python写一个计算斐波那契数列的函数',
        '解释一下"光合作用"的基本过程',
        '如果明天下雨，我应该如何出门',
        '比较一下猫和狗作为宠物的优缺点',
        '解释什么是机器学习',
        '推荐一些中国的美食'
    ]
    
    # 消息历史列表，存储格式: [{"role": "user", "content": "..."}]
    conversation = []
    
    # 初始化模型与分词器
    model, tokenizer = init_model(args)
    
    # 选择交互模式
    input_mode = int(input('[0] 自动测试 (预设 Prompts)\n[1] 手动对话 (交互模式)\n输入编号: '))
    
    # 实例化流式输出器，实现在生成过程中逐 Token 打印
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 构造输入迭代器
    prompt_iter = test_prompts if input_mode == 0 else iter(lambda: input('💬 用户: '), '')
    
    for prompt in prompt_iter:
        # 设置随机种子确保生成结果的多样性
        setup_seed(random.randint(0, 31415926))
        
        if input_mode == 0: 
            print(f'💬 用户: {prompt}')
            
        # 1. 对话历史管理：根据 args.historys 截断旧的消息
        conversation = conversation[-args.historys:] if args.historys else []
        conversation.append({"role": "user", "content": prompt})
        
        # 2. 构造模型输入 Prompt
        if 'pretrain' in args.weight:
            # 预训练模型通常不带对话模板，直接拼接
            inputs_text = tokenizer.bos_token + prompt
        else:
            # 指令微调模型应用 chat_template 转换为特定格式的字符串
            inputs_text = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True, 
                open_thinking=bool(args.open_thinking)
            )
        
        # 3. Tokenize：[text] -> [1, seq_len] Tensor
        inputs = tokenizer(inputs_text, return_tensors="pt", truncation=True).to(args.device)

        print('🧠 MiniMind: ', end='')
        st = time.time()
        
        # 4. 执行自回归生成
        generated_ids = model.generate(
            inputs=inputs["input_ids"], 
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens, 
            do_sample=True, 
            streamer=streamer, # 启用流式打印
            pad_token_id=tokenizer.pad_token_id, 
            eos_token_id=tokenizer.eos_token_id,
            top_p=args.top_p, 
            temperature=args.temperature, 
            repetition_penalty=1.0 # 重复惩罚系数
        )
        
        # 5. 提取生成部分的文本并存入历史
        # 注意：generated_ids 包含原始输入的 Prompt，需要切片提取
        response_ids = generated_ids[0][len(inputs["input_ids"][0]):]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        conversation.append({"role": "assistant", "content": response_text})
        
        # 6. 计算推理速度指标
        gen_tokens_count = len(response_ids)
        duration = time.time() - st
        if args.show_speed:
            print(f'\n[推理速度]: {gen_tokens_count / duration:.2f} tokens/s\n\n')
        else:
            print('\n\n')

if __name__ == "__main__":
    main()
