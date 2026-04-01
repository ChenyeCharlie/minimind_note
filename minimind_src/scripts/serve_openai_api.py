import argparse
import json
import re
import os
import sys

# 设置当前包名为 scripts，并添加上级目录路径以支持导入 model 模块
__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

# 忽略不必要的警告
warnings.filterwarnings('ignore')

# 实例化 FastAPI 应用
app = FastAPI()


def init_model(args):
    """
    模型初始化：加载分词器和 MiniMind 模型（支持原生 pth 和 Transformers 格式）。
    """
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    
    if 'model' in args.load_from:
        # 加载原生 pth 权重
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        
        # 构造 MiniMind 配置
        config = MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        )
        model = MiniMindForCausalLM(config)
        # 加载权重到模型
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        
        # 处理 LoRA 权重加载
        if args.lora_weight != 'None':
            apply_lora(model)
            load_lora(model, f'../{args.save_dir}/lora/{args.lora_weight}_{args.hidden_size}.pth')
    else:
        # 加载标准的 HuggingFace 模型格式
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
        
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    # 使用半精度推理以提高效率
    return model.half().eval().to(device), tokenizer


class ChatRequest(BaseModel):
    """
    定义 OpenAI 兼容的聊天请求数据模型。
    """
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = []
    open_thinking: bool = False
    chat_template_kwargs: dict = None
    
    def get_open_thinking(self) -> bool:
        """
        统一解析开启思考链（thinking）的配置方式。
        """
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    """
    自定义流式处理器：将生成的文本 Token 推送到队列中，供异步 API 读取。
    """
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        """当一个或多个 Token 被解码为文本时触发。"""
        self.queue.put(text)
        if stream_end:
            # 标记流结束
            self.queue.put(None)


def parse_response(text):
    """
    后处理模型生成的文本：
    1. 提取 <think> 标签内的思考内容。
    2. 提取 <tool_call> 标签内的 JSON 工具调用。
    3. 返回清洗后的文本内容。
    """
    reasoning_content = None
    # 尝试匹配 <think>...</think>
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        # 处理不完整的闭合情况
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
        
    # 解析工具调用
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({
                "id": f"call_{int(time.time())}_{i}",
                "type": "function",
                "function": {
                    "name": call.get("name", ""),
                    "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)
                }
            })
        except Exception:
            pass
            
    # 从最终文本中移除工具调用标签内容
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        
    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    """
    流式生成生成器：
    异步运行模型生成逻辑，并将结果封装为 OpenAI Server-Sent Events (SSE) 格式。
    """
    try:
        # 1. 应用对话模板，限制输入长度
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)[-max_tokens:]
        # 2. Tokenize [text] -> [1, seq_len]
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        # 3. 在子线程中启动模型生成
        def _generate():
            model.generate(
                inputs.input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                attention_mask=inputs.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer
            )

        Thread(target=_generate).start()

        full_text = ""
        emitted = 0
        thinking_ended = not bool(open_thinking)

        # 4. 循环从队列中读取生成的文本并 Yield
        while True:
            text = queue.get()
            if text is None: # 生成结束
                break
            full_text += text

            # 分段逻辑：优先发送思考内容 (reasoning_content)，检测到 </think> 后切换到正式内容 (content)
            if not thinking_ended:
                pos = full_text.find('</think>')
                if pos >= 0:
                    thinking_ended = True
                    new_r = full_text[emitted:pos]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                    emitted = pos + len('</think>')
                    after = full_text[emitted:].lstrip('\n')
                    emitted = len(full_text) - len(after)
                    if after:
                        yield json.dumps({"choices": [{"delta": {"content": after}}]}, ensure_ascii=False)
                        emitted = len(full_text)
                else:
                    new_r = full_text[emitted:]
                    if new_r:
                        yield json.dumps({"choices": [{"delta": {"reasoning_content": new_r}}]}, ensure_ascii=False)
                        emitted = len(full_text)
            else:
                new_c = full_text[emitted:]
                if new_c:
                    yield json.dumps({"choices": [{"delta": {"content": new_c}}]}, ensure_ascii=False)
                    emitted = len(full_text)

        # 5. 最后检查是否存在工具调用并发送
        _, _, tool_calls = parse_response(full_text)
        if tool_calls:
            yield json.dumps({"choices": [{"delta": {"tool_calls": tool_calls}}]}, ensure_ascii=False)
        # 发送结束标记
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_calls else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    OpenAI 核心对话接口端点。
    """
    try:
        if request.stream:
            # 流式回复模式
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            # 非流式（同步）回复模式
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )[-request.max_tokens:]
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_length=inputs["input_ids"].shape[1] + request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                # 解码回复部分
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            
            # 解析内容、思考和工具调用
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
                
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind OpenAI 兼容服务后端")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径")
    parser.add_argument('--save_dir', default='out', type=str, help="权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重命名前缀")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重前缀")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="层数")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大截断长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否为MoE")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE外推")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="设备")
    
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    
    # 启动 uvicorn 服务
    uvicorn.run(app, host="0.0.0.0", port=8998)
