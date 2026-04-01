import os
import sys
import json

# 设置当前包名为 scripts，并添加上级目录路径以支持导入 model 模块
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

# 忽略 UserWarning 警告（通常是版本兼容性提示）
warnings.filterwarnings('ignore', category=UserWarning)

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    """
    将原生 PyTorch (.pth) 权重转换为自定义的 Transformers-MiniMind 格式。
    此格式使用了 MiniMind 专属的模型类，需要在加载时信任远程代码（trust_remote_code=True）。
    """
    # 1. 注册自定义类到 AutoClass，使其支持 save_pretrained 自动生成加载代码
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    
    # 2. 初始化模型并加载权重
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    
    # 3. 转换模型精度（通常转为 float16 以适配生态）
    lm_model = lm_model.to(dtype)
    
    # 4. 统计参数量
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    
    # 5. 保存模型权重和分词器
    # safe_serialization=False 表示保存为 .bin 格式而非 .safetensors（MiniMind 默认习惯）
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    
    # 6. 针对新版本 Transformers 的兼容性修正
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path = os.path.join(transformers_path, "tokenizer_config.json")
        config_path = os.path.join(transformers_path, "config.json")
        
        # 修正分词器类名
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        
        # 修正 RoPE 相关的配置键名
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    """
    将原生 PyTorch (.pth) 权重转换为通用的生态结构（如 Qwen3 格式）。
    这使得模型可以无需自定义代码即可被主流推理框架（如 vLLM, Ollama）识别。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    
    # 1. 构造与原生 MiniMind 对应的标准生态配置
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": True # 权重共享开关
    }
    
    # 2. 根据是否使用 MoE 映射到不同的 Qwen 结构
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(**common_config, use_sliding_window=False, sliding_window=None)
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        
        # 3. 针对 MoE 的特殊权重重排（针对生态兼容性）
        # 生态中的 MoE 权重通常将 gate 和 up 投影拼接在同一矩阵中（gate_up_proj）
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                # 拼接 gate 和 up 投影: [num_experts, hidden, inter] -> [num_experts, hidden, 2*inter]
                new_sd[f'{p}.gate_up_proj'] = torch.cat([
                    torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), 
                    torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])
                ], dim=1)
                # 提取 down 投影
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    # 4. 加载并保存
    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype).save_pretrained(transformers_path)
    
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # 5. 后处理修正（同上）
    if int(transformers.__version__.split('.')[0]) >= 5:
        # ... (修正 config.json 和 tokenizer_config.json 的逻辑同 convert_torch2transformers_minimind)
        pass
        
    print(f"模型已保存为通用 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    """将 Transformers 格式模型转回原生 PyTorch 字典格式。"""
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    # 转为 CPU 半精度以节省空间
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    """
    核心转换功能：将 LoRA 旁路权重永久合并进基模权重中。
    1. 加载基模。
    2. 应用 LoRA 结构。
    3. 调用 model_lora.py 的 merge_lora 执行权重加法并保存。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    
    # 注入并合并
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为全量基模结构: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    """将 Jinja2 模板文件转换为 JSON 转义字符串，方便粘贴进 tokenizer_config.json。"""
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    """从 tokenizer_config.json 中提取 chat_template 并保存为独立的 .jinja 文件进行编辑。"""
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    # 配置目标模型的结构参数
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=True)
    
    # 示例用法：将 PTH 权重转为 Transformers 格式
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3-moe'
    convert_torch2transformers(torch_path, transformers_path)

    # 更多功能可以通过取消下方注释来运行：
    # 1. 合并 LoRA 权重
    # ... 
    # 2. 模板文件互转
    # ...
