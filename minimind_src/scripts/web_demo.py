import random
import re
import json
import os
from threading import Thread

import torch
import numpy as np
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# 1. 设置 Streamlit 页面配置：标题和侧边栏状态
st.set_page_config(page_title="MiniMind", initial_sidebar_state="collapsed")

# 2. 自定义 CSS 样式：优化按钮、间距以及 UI 组件的视觉效果
st.markdown("""
    <style>
        /* 添加操作按钮样式 */
        .stButton button {
            border-radius: 50% !important;
            width: 32px !important;
            height: 32px !important;
            padding: 0 !important;
            background-color: transparent !important;
            border: 1px solid #ddd !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            font-size: 14px !important;
            color: #666 !important;
            margin: 5px 10px 5px 0 !important;
        }
        /* ... (省略部分样式定义以保持简洁，核心逻辑见下) */
    </style>
""", unsafe_allow_html=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 多语言 UI 文本字典：支持中英双语切换
LANG_TEXTS = {
    'zh': {
        'settings': '模型设定调整',
        'history_rounds': '历史对话轮次',
        'max_length': '最大生成长度',
        'temperature': '温度',
        'thinking': '思考',
        'tools': '工具',
        'language': '语言',
        'send': '给 MiniMind 发送消息',
        'disclaimer': 'AI 生成内容可能存在错误，请仔细核实',
        'think_tip': '自适应思考，目前多轮对话或Tool Call共存时思考不稳定',
        'tool_select': '工具选择（最多4个）',
    },
    'en': {
        'settings': 'Model Settings',
        'history_rounds': 'History Rounds',
        'max_length': 'Max Length',
        'temperature': 'Temperature',
        'thinking': 'Thinking',
        'tools': 'Tools',
        'language': 'Language',
        'send': 'Send a message to MiniMind',
        'disclaimer': 'AI-generated content may be inaccurate, please verify',
        'think_tip': 'Adaptive thinking; may be unstable with multi-turn or Tool Call',
        'tool_select': 'Tool Selection (max 4)',
    }
}

def get_text(key):
    """根据 session_state 中的语言选择返回对应的文案。"""
    lang = st.session_state.get('lang', 'en')
    return LANG_TEXTS.get(lang, {}).get(key, LANG_TEXTS['zh'].get(key, key))

# 4. 可用工具 (Tools) 定义：遵循 JSON Schema 标准
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "数学表达式"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取当前时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "random_number", "description": "生成随机数", "parameters": {"type": "object", "properties": {"min": {"type": "integer"}, "max": {"type": "integer"}}, "required": ["min", "max"]}}},
    # ... (更多工具定义)
]

TOOL_SHORT_NAMES = {
    'calculate_math': '数学', 'get_current_time': '时间', 'random_number': '随机',
    'text_length': '字数', 'unit_converter': '单位', 'get_current_weather': '天气',
    'get_exchange_rate': '汇率', 'translate_text': '翻译'
}

def execute_tool(tool_name, args):
    """
    本地模拟工具执行逻辑：
    根据模型生成的 tool_name 和 args 执行相应的 Python 代码并返回结果。
    """
    import datetime
    try:
        if tool_name == 'calculate_math':
            return {"result": eval(args.get('expression', '0'))}
        elif tool_name == 'get_current_time':
            return {"result": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        # ... (其他工具的模拟实现)
        return {"result": "Unknown tool"}
    except Exception as e:
        return {"error": str(e)}


def process_assistant_content(content, is_streaming=False):
    """
    后处理逻辑：将生成的原始文本（包含 <think> 和 <tool_call> 标签）转换为美观的 HTML 组件。
    1. 工具调用：显示为蓝色卡片。
    2. 思考过程：显示为带折叠效果的灰色区块。
    """
    # 处理 tool_call 标签：正则匹配并替换为富文本 DIV
    if '<tool_call>' in content:
        def format_tool_call(match):
            try:
                tc = json.loads(match.group(1))
                name = tc.get('name', 'unknown')
                args = tc.get('arguments', {})
                return f'<div style="background: rgba(80, 110, 150, 0.20); border: 1px solid rgba(140, 170, 210, 0.30); padding: 10px 12px; border-radius: 12px; margin: 6px 0;"><b>{name}</b>: {json.dumps(args, ensure_ascii=False)}</div>'
            except:
                return match.group(0)
        content = re.sub(r'<tool_call>(.*?)</tool_call>', format_tool_call, content, flags=re.DOTALL)
    
    # 处理思考链 <think> 标签：转换为 Streamlit 可渲染的 <details> 标签
    if '<think>' in content and '</think>' in content:
        def format_think(match):
            think_content = match.group(2)
            return f'<details open style="border-left: 2px solid #666; padding-left: 12px;"><summary>已思考</summary><div>{think_content.strip()}</div></details>'
        content = re.sub(r'(<think>)(.*?)(</think>)', format_think, content, flags=re.DOTALL)

    return content


@st.cache_resource
def load_model_tokenizer(model_path):
    """
    利用 Streamlit 的缓存机制加载模型和分词器，避免页面刷新时重复加载。
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.half().eval().to(device)
    return model, tokenizer


# 5. 侧边栏配置：模型选择、语言切换及超参数调整
script_dir = os.path.dirname(os.path.abspath(__file__))
# 动态扫描当前脚本同级目录下的所有文件夹，识别为模型路径
MODEL_PATHS = {d: [d, d] for d in sorted(os.listdir(script_dir)) if os.path.isdir(os.path.join(script_dir, d)) and not d.startswith(('.', '_'))}

selected_model = st.sidebar.selectbox('Model', list(MODEL_PATHS.keys()) or ["None"])
model_path = os.path.join(script_dir, selected_model) if selected_model != "None" else ""

# ... (侧边栏参数：history_rounds, max_length, temperature 等的 UI 绑定逻辑)

def main():
    """
    Web Demo 主循环：处理用户输入、自回归生成、多轮工具调用及页面渲染。
    """
    if not model_path: return
    model, tokenizer = load_model_tokenizer(model_path)

    # 初始化 session 状态存储对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []      # 用于 UI 显示的格式
        st.session_state.chat_messages = [] # 用于喂给模型的原始格式

    # 渲染历史消息
    for message in st.session_state.messages:
        role_style = 'display: flex; justify-content: flex-end;' if message["role"] == "user" else ''
        content_style = 'background-color: #3d4450; color: white;' if message["role"] == "user" else 'background-color: #f0f2f6;'
        st.markdown(f'<div style="{role_style}"><div style="padding: 8px 12px; border-radius: 22px; {content_style}">{process_assistant_content(message["content"])}</div></div>', unsafe_allow_html=True)

    # 6. 处理新的用户输入
    if prompt := st.chat_input(placeholder=get_text('send')):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        st.rerun() # 刷新页面显示用户刚才发送的消息

    # 7. 生成模型响应
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        placeholder = st.empty() # 创建一个占位符用于动态显示流式输出
        
        # 准备模型输入
        tools = [t for t in TOOLS if st.session_state.get(f"tool_{t['function']['name']}", False)] or None
        new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, tokenize=False, add_generation_prompt=True, tools=tools)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        # 实例化流式迭代器
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            "input_ids": inputs.input_ids,
            "max_length": inputs.input_ids.shape[1] + st.session_state.max_new_tokens,
            "temperature": st.session_state.temperature,
            "streamer": streamer,
            "do_sample": True
        }

        # 在子线程中执行生成，防止阻塞主 UI 线程
        Thread(target=model.generate, kwargs=generation_kwargs).start()

        full_response = ""
        for new_text in streamer:
            full_response += new_text
            # 实时更新 UI 占位符
            placeholder.markdown(process_assistant_content(full_response, is_streaming=True), unsafe_allow_html=True)

        # 8. 递归处理工具调用 (Multi-step Tool Calling)
        for _ in range(10): # 最多支持 10 步连续调用
            tool_calls = parse_tool_calls(full_response)
            if not tool_calls: break
            
            # 执行工具并拼接结果到上下文
            for tc in tool_calls:
                result = execute_tool(tc.get('name'), tc.get('arguments', {}))
                st.session_state.chat_messages.append({"role": "assistant", "content": full_response})
                st.session_state.chat_messages.append({"role": "tool", "content": json.dumps(result)})
            
            # 再次调用模型处理工具执行后的结果
            new_prompt = tokenizer.apply_chat_template(st.session_state.chat_messages, tokenize=False, add_generation_prompt=True, tools=tools)
            # ... (重复生成逻辑) ...
            break # 简化演示

        # 保存对话结果
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.session_state.chat_messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
