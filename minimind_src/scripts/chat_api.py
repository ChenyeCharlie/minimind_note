from openai import OpenAI

# 1. 初始化 OpenAI 客户端，连接到本地运行的服务 (如 Ollama 或 serve_openai_api.py 启动的服务)
client = OpenAI(
    api_key="sk-123", # 占位符 API Key
    base_url="http://localhost:11434/v1" # 默认连接到本地 11434 端口的 V1 接口
)

# 是否启用流式输出 (逐字显示)
stream = True
# 初始对话历史
conversation_history_origin = []
conversation_history = conversation_history_origin.copy()
# 历史消息携带数量：必须为偶数 (包含提问和回答)，设为 0 则只发送当前提问
history_messages_num = 0  

while True:
    # 2. 获取用户输入
    query = input('[Q]: ')
    if not query: continue
    
    # 3. 将用户提问存入对话历史
    conversation_history.append({"role": "user", "content": query})
    
    # 4. 调用 Chat Completion 接口
    # 参数说明：
    # model: 模型名称
    # messages: 截取最近几轮对话历史
    # stream: 流式输出开关
    # extra_body: 包含 MiniMind 特有的思考链控制参数 (open_thinking)
    response = client.chat.completions.create(
        model="minimind-local:latest",
        messages=conversation_history[-(history_messages_num or 1):],
        stream=stream,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.8,
        extra_body={
            "chat_template_kwargs": {"open_thinking": True}, # 开启思考链显示
            "reasoning_effort": "medium" # 思考程度 (针对特定后端)
        }
    )

    if not stream:
        # 5a. 非流式处理：一次性获取完整回复
        assistant_res = response.choices[0].message.content
        print('[A]: ', assistant_res)
    else:
        # 5b. 流式处理：实时打印生成内容
        print('[A]: ', end='', flush=True)
        assistant_res = ''
        for chunk in response:
            delta = chunk.choices[0].delta
            # 尝试提取思考内容 (reasoning_content) 和 正式内容 (content)
            # 思考内容通常显示为灰色 (ANSI 编码 \033[90m)
            r = getattr(delta, 'reasoning_content', None) or ""
            c = delta.content or ""
            
            if r:
                print(f'\033[90m{r}\033[0m', end="", flush=True)
            if c:
                print(c, end="", flush=True)
            assistant_res += c

    # 6. 将模型回复存入对话历史，以便下一轮迭代
    conversation_history.append({"role": "assistant", "content": assistant_res})
    print('\n\n')
