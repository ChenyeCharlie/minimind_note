import os
import sys

# 设置当前包名为 scripts，并添加上级目录路径以支持导入 model 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import json
import time
import random
import argparse
import warnings
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from openai import OpenAI
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed, get_model_params

# 忽略不必要的警告信息
warnings.filterwarnings('ignore')

# 1. 定义可用的工具集合 (Following OpenAI Tool Call Format)
# 每个工具包含：类型、函数名、功能描述以及参数的 JSON Schema 结构
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式的结果，支持加减乘除、幂运算、开方等", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式，如123+456、2**10、sqrt(144)"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前日期和时间，支持指定时区", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "description": "时区名称，如Asia/Shanghai、America/New_York", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成指定范围内的随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer", "description": "最小值", "default": 0}, "max": {"type": "integer", "description": "最大值", "default": 100}}, "required": []}}},
    {"type": "function", "function": {"name": "text_length", "description": "计算文本的字符数和单词数", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要统计的文本"}}, "required": ["text"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "进行单位换算，支持长度、重量、温度等", "parameters": {"type": "object", "properties": {"value": {"type": "number", "description": "要转换的数值"}, "from_unit": {"type": "string", "description": "源单位，如km、miles、kg、pounds、celsius、fahrenheit"}, "to_unit": {"type": "string", "description": "目标单位"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取指定城市的当前天气信息，包括温度、湿度和天气状况", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "城市名称，如北京、上海、New York"}, "unit": {"type": "string", "description": "温度单位，celsius或fahrenheit", "enum": ["celsius", "fahrenheit"], "default": "celsius"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询两种货币之间的实时汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string", "description": "源货币代码，如USD、CNY、EUR"}, "to_currency": {"type": "string", "description": "目标货币代码，如USD、CNY、EUR"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "将文本翻译成目标语言", "parameters": {"type": "object", "properties": {"text": {"type": "string", "description": "要翻译的文本"}, "target_language": {"type": "string", "description": "目标语言，如english、chinese、japanese、french"}}, "required": ["text", "target_language"]}}},
]

# 2. 模拟工具执行的结果映射
# 使用 lambda 表达式定义每个工具的逻辑实现
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("²", "**2").replace("³", "**3").replace("（", "(").replace("）", ")")))},
    "get_current_time": lambda args: {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "random_number": lambda args: {"result": random.randint(int(args.get("min", 0)), int(args.get("max", 100)))},
    "text_length": lambda args: {"characters": len(args.get("text", "")), "words": len(args.get("text", "").split())},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * 0.621371, 2), "from": f"{args.get('value', 0)} {args.get('from_unit', '')}", "to": args.get("to_unit", "")},
    "get_current_weather": lambda args: {"city": args.get("location"), "temperature": "22°C", "humidity": "65%", "condition": "晴"},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency", ""), "to": args.get("to_currency", ""), "rate": 7.15},
    "translate_text": lambda args: {"translated": "hello world"},
}

# 构造工具名到定义的映射，方便快速查找
TOOL_MAP = {t["function"]["name"]: t for t in TOOLS}

def get_tools(names):
    """根据名称列表获取对应的工具定义。"""
    return [TOOL_MAP[n] for n in names]

# 预设的测试用例：包含用户提问和允许模型调用的工具子集
TEST_CASES = [
    {"prompt": "帮我算一下 256 乘以 37 等于多少", "tools": ["calculate_math", "get_current_time"]},
    {"prompt": "现在几点了？", "tools": ["get_current_time", "random_number"]},
    {"prompt": "帮我把100公里换算成英里", "tools": ["unit_converter", "calculate_math"]},
    {"prompt": "帮我生成一个1到1000的随机数，然后计算它的平方", "tools": ["random_number", "calculate_math", "text_length"]},
    {"prompt": "北京今天天气怎么样？", "tools": ["get_current_weather", "get_current_time"]},
    {"prompt": "查一下美元兑人民币汇率", "tools": ["get_exchange_rate", "get_current_time"]},
    {"prompt": "把'你好世界'翻译成英文", "tools": ["translate_text", "text_length"]},
    {"prompt": "What is the weather in Tokyo? Also convert 30 celsius to fahrenheit.", "tools": ["get_current_weather", "unit_converter", "get_current_time"]},
]


def init_model(args):
    """
    加载本地 MiniMind 模型和分词器。
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if 'model' in args.load_from:
        # 加载原生训练的 pth 权重
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size, 
            num_hidden_layers=args.num_hidden_layers, 
            use_moe=bool(args.use_moe)
        ))
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'./{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model.load_state_dict(torch.load(ckp, map_location=args.device), strict=True)
    else:
        # 加载 Transformers 格式权重
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    
    get_model_params(model, model.config)
    return model.half().eval().to(args.device), tokenizer


def parse_tool_calls(text):
    """
    从模型生成的原始文本中利用正则匹配提取 <tool_call> 标签内的 JSON 内容。
    用于本地模型推理后处理。
    """
    matches = re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    calls = []
    for m in matches:
        try:
            calls.append(json.loads(m.strip()))
        except Exception:
            pass
    return calls


def parse_tool_call_from_text(content):
    """
    从回复内容中提取工具调用，并封装为 OpenAI 兼容的格式。
    用于 API 模式下，当接口没有直接返回 tool_calls 字段时的降级解析。
    """
    pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    matches = re.findall(pattern, content, re.DOTALL)
    if not matches:
        return None
    tool_calls = []
    for i, match in enumerate(matches):
        try:
            data = json.loads(match)
            tool_calls.append({
                "id": f"call_{i}",
                "function": {
                    "name": data.get("name", ""), 
                    "arguments": json.dumps(data.get("arguments", {}), ensure_ascii=False)
                }
            })
        except Exception:
            pass
    return tool_calls if tool_calls else None


def execute_tool(call, arguments=None):
    """
    工具执行分发器：
    解析参数并调用 MOCK_RESULTS 中对应的逻辑实现。
    """
    name = call.get("name", "") if isinstance(call, dict) else call
    try:
        raw_args = call.get("arguments", {}) if isinstance(call, dict) else arguments
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        args = {}
        
    fn = MOCK_RESULTS.get(name)
    if not fn:
        return {"error": f"未知工具: {name}"}
    try:
        return fn(args)
    except Exception as e:
        return {"error": f"工具执行失败: {str(e)[:80]}"}


def generate(model, tokenizer, messages, tools, args):
    """
    本地生成函数：
    1. 应用对话模板拼接上下文。
    2. 执行推理并实时显示输出。
    """
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 构造包含工具定义的输入文本
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=False)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(args.device)
    
    st = time.time()
    print('🧠: ', end='')
    # 生成 Token
    generated_ids = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"],
        max_new_tokens=args.max_new_tokens, do_sample=True, streamer=streamer,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        top_p=args.top_p, temperature=args.temperature
    )
    # 解码生成结果 (排除输入部分)
    response = tokenizer.decode(generated_ids[0][len(inputs["input_ids"][0]):], skip_special_tokens=True)
    
    gen_tokens = len(generated_ids[0]) - len(inputs["input_ids"][0])
    if args.show_speed:
        print(f'\n[Speed]: {gen_tokens / (time.time() - st):.2f} tokens/s')
    else:
        print()
    return response


def chat_api(client, messages, tools, args, stream=True):
    """
    API 生成函数：调用 OpenAI 兼容接口进行推理。
    支持流式解析 content 和 tool_calls。
    """
    response = client.chat.completions.create(
        model=args.api_model, messages=messages, tools=tools,
        stream=stream, temperature=args.temperature,
        max_tokens=8192, top_p=args.top_p
    )
    
    if not stream:
        # 非流式解析
        choice = response.choices[0]
        content = choice.message.content or ""
        tool_calls = choice.message.tool_calls
        if not tool_calls:
            tool_calls = parse_tool_call_from_text(content)
        print(f'🧠: {content}')
        return content, tool_calls
        
    # 流式解析
    print('🧠: ', end='', flush=True)
    content, tool_calls = "", None
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            if tool_calls is None: tool_calls = []
            for tc_chunk in delta.tool_calls:
                # 针对流式片段中的 tool_call 进行 ID、函数名和参数的累加拼接
                idx = tc_chunk.index if tc_chunk.index is not None else len(tool_calls)
                while len(tool_calls) <= idx:
                    tool_calls.append({"id": "", "function": {"name": "", "arguments": ""}})
                if tc_chunk.id:
                    tool_calls[idx]["id"] += tc_chunk.id
                if tc_chunk.function:
                    if tc_chunk.function.name:
                        tool_calls[idx]["function"]["name"] += tc_chunk.function.name
                    if tc_chunk.function.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc_chunk.function.arguments
    print()
    # 如果接口返回的 tool_calls 字段为空，尝试从文本内容中正则提取
    if not tool_calls:
        tool_calls = parse_tool_call_from_text(content)
    return content, tool_calls


def run_case(prompt, tools, args, model=None, tokenizer=None, client=None):
    """
    单测试用例执行闭环：
    支持多轮交互（模型输出 -> 解析工具调用 -> 执行工具 -> 反馈结果 -> 模型再次输出）。
    """
    messages = [{"role": "user", "content": prompt}]
    while True:
        # 1. 模型推理
        if args.backend == 'local':
            content = generate(model, tokenizer, messages, tools, args)
            tool_calls = parse_tool_calls(content)
        else:
            content, tool_calls = chat_api(client, messages, tools, args, stream=bool(args.stream))
            
        if not tool_calls:
            # 没有工具调用意图，对话正常结束
            break
            
        # 2. 格式化工具调用信息，存入对话历史
        tool_calls_formatted = []
        for tc in tool_calls:
            tc_info = {
                "id": tc.id if hasattr(tc, 'id') else tc.get("id", ""),
                "name": tc.function.name if hasattr(tc, 'function') else tc["function"]["name"],
                "arguments": tc.function.arguments if hasattr(tc, 'function') else tc["function"]["arguments"]
            }
            tool_calls_formatted.append(tc_info)
            
        messages.append({
            "role": "assistant", 
            "content": content, 
            "tool_calls": [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in tool_calls_formatted]
        } if args.backend == 'api' else {"role": "assistant", "content": content})
        
        # 3. 逐个执行工具调用并获取反馈 (Observation)
        for tc in tool_calls_formatted:
            name = tc["name"]
            arguments = tc["arguments"]
            print(f'📞 [Tool Calling]: {name} | args={arguments}')
            
            # 执行本地模拟逻辑
            result = execute_tool(tc if args.backend == 'local' else name, arguments)
            print(f'✅ [Tool Called]: {json.dumps(result, ensure_ascii=False)}')
            
            # 将工具执行结果存入对话历史
            messages.append({
                "role": "tool", 
                "content": json.dumps(result, ensure_ascii=False), 
                "tool_call_id": tc["id"]
            } if args.backend == 'api' else {
                "role": "tool", 
                "content": json.dumps(result, ensure_ascii=False)
            })
        # 4. 继续循环，让模型基于工具返回的结果进行下一轮思考或最终回答


def main():
    parser = argparse.ArgumentParser(description="MiniMind Tool 调用能力评测脚本")
    parser.add_argument('--backend', default='local', choices=['local', 'api'], type=str, help="推理后端选择")
    parser.add_argument('--load_from', default='../model', type=str, help="模型路径")
    parser.add_argument('--save_dir', default='../out', type=str, help="权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="pth前缀")
    parser.add_argument('--hidden_size', default=768, type=int, help="维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="MoE")
    parser.add_argument('--max_new_tokens', default=512, type=int, help="长度")
    parser.add_argument('--temperature', default=0.9, type=float, help="温度")
    parser.add_argument('--top_p', default=0.9, type=float, help="核采样")
    parser.add_argument('--show_speed', default=0, type=int, help="速度显示")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="设备")
    parser.add_argument('--api_base_url', default="http://localhost:11434/v1", type=str, help="API URL")
    parser.add_argument('--api_key', default='sk-123', type=str, help="API Key")
    parser.add_argument('--api_model', default='jingyaogong/minimind-3:latest', type=str, help="API Model")
    parser.add_argument('--stream', default=1, type=int, help="流式")
    args = parser.parse_args()

    # 初始化后端
    model = tokenizer = client = None
    if args.backend == 'local': 
        model, tokenizer = init_model(args)
    else: 
        client = OpenAI(api_key=args.api_key, base_url=args.api_base_url)

    # 模式选择
    input_mode = int(input('[0] 自动批量测试\n[1] 手动输入测试\n输入编号: '))

    # 构造测试队列
    if input_mode == 0:
        cases = [{"prompt": case["prompt"], "tools": get_tools(case["tools"]), "tool_names": case["tools"]} for case in TEST_CASES]
    else:
        cases = iter(lambda: {"prompt": input('💬 用户输入: '), "tools": TOOLS, "tool_names": [t["function"]["name"] for t in TOOLS]}, {"prompt": ""})
        
    for case in cases:
        if not case["prompt"]: break
        setup_seed(random.randint(0, 31415926))
        if input_mode == 0:
            print(f'📦 当前可用工具: {case["tool_names"]}\n')
            print(f'💬 用户: {case["prompt"]}')
            
        # 执行完整的 Tool Call 交互流程
        run_case(case["prompt"], case["tools"], args, model=model, tokenizer=tokenizer, client=client)
        print('\n' + '-' * 50 + '\n')


if __name__ == "__main__":
    main()
