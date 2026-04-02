from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value

# 设置环境变量，关闭分词器的并行处理以避免死锁或警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """
    对话数据的预处理逻辑：
    1. 如果对话中包含 tools（工具调用），则不做处理，保持数据原样。
    2. 否则，有一定概率（add_system_ratio）随机添加预设的 System Prompt。
    """
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): 
        return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    
    # 如果第一条消息不是 system 角色，则按概率插入 system 消息
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """
    对话内容后处理：
    按概率（1 - empty_think_ratio）移除模型生成中可能出现的空思考标签 <think>\n\n</think>\n\n。
    """
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """
    预训练数据集类：
    用于无监督的 Next Token Prediction 训练。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 使用 Huggingface datasets 加载本地 JSON 格式数据集
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # 1. 读取文本内容
        sample = self.samples[index]
        # 2. Tokenize 文本: [text] -> [ids]
        # 预留 2 个位置给 BOS 和 EOS 标记
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        
        # 3. 构造完整序列: [BOS] + tokens + [EOS]
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        
        # 4. Padding: 如果长度不足，用 pad_token 补齐到 max_length
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        # 5. 构造张量 [max_length]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 6. 构造 Label: 预训练中 Label 等于 Input，但 Padding 部分设为 -100 以在交叉熵中忽略
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return input_ids, labels

class SFTDataset(Dataset):
    """
    指令微调 (SFT) 数据集类：
    解析多轮对话，并对 labels 进行掩码，只计算 Assistant 回复部分的 Loss。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 定义数据集特征字段
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        
        # 预计算 assistant 的开始和结束标记序列，用于在 labels 中定位回复部分
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        """
        根据对话历史和工具定义，利用 tokenizer 的 chat_template 生成训练用的字符串。
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 提取系统提示词中的工具定义
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # 解析工具调用信息
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
            
        # 调用模板转换，生成类似 "<bos>user\n...<eos>\n<bos>assistant\n..." 的长文本
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        """
        生成 Labels:
        - 默认全部设为 -100 (不计算 Loss)。
        - 寻找到 assistant 的回复区间 [start, end]，将该区间内的 token id 填入 labels。
        """
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 匹配到 assistant 的起始标记
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 寻找对应的结束标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # 将 Assistant 回复部分 (包含结束标记) 设为可计算 Loss
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 1. 预处理对话 (添加 system prompt)
        conversations = pre_processing_chat(sample['conversations'])
        # 2. 生成模板化字符串
        prompt = self.create_chat_prompt(conversations)
        # 3. 后处理内容 (如移除空思考标签)
        prompt = post_processing_chat(prompt)
        # 4. 转换成 input_ids: [seq_len] -> [max_length]
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        # 5. 生成掩码后的 labels
        labels = self.generate_labels(input_ids)
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

class DPODataset(Dataset):
    """
    DPO (Direct Preference Optimization) 数据集类：
    加载 (chosen, rejected) 偏好对数据。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 同样需要定位 Assistant 的回复区间
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen = sample['chosen']      # 更好的回答列表
        rejected = sample['rejected']  # 较差的回答列表
        
        # 1. 为两个回复分别应用模板和后处理
        chosen_prompt = self.tokenizer.apply_chat_template(chosen, tokenize=False, add_generation_prompt=False)
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(rejected, tokenize=False, add_generation_prompt=False)
        rejected_prompt = post_processing_chat(rejected_prompt)
        
        # 2. Tokenize 两个序列
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')

        chosen_input_ids = chosen_encoding['input_ids']
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        
        # 3. 构造 DPO 训练需要的张量对 (x, y, mask)
        # x: [0...N-1], y: [1...N], mask: [1...N]
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        """
        DPO 同样只对回复部分计算概率，其余部分 mask=0
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

class RLAIFDataset(Dataset):
    """
    RLAIF 数据集类：
    用于强化学习在线采样，只返回 Prompt 部分，不包含回复。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.thinking_ratio = thinking_ratio  # 控制是否开启模型思考链
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 仅取对话历史部分，准备让模型生成最新的回答
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )
        
    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        return {
            'prompt': prompt,
            'answer': ""
        }

class AgentRLDataset(Dataset):
    """
    Agent 强化学习数据集类：
    专门处理包含工具调用信息的任务，保留 Ground Truth (gt) 用于对比。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        """
        解析对话历史中的消息和工具定义。
        """
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # 提取系统提示词中的 tools 字段
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # 返回对话历史（不含最后一条回复）和工具集
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        return {'messages': messages, 'tools': tools, 'gt': sample['gt']}

if __name__ == "__main__":
    pass
