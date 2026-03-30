## minimind 项目速览

`minimind` 是一个面向训练与对齐的自研 LLM 工程：核心模型 `MiniMindForCausalLM`（支持 MoE、RoPE、GQA）配套 LoRA；训练侧覆盖 Pretrain/SFT/LoRA、DPO、知识蒸馏、以及 PPO/GRPO/AgentRL 等强化学习变体；推理侧提供本地生成与 OpenAI 兼容服务，并支持 `<think>` 与 `<tool_call>` 的解析与工具回填。

---

## 目录结构与核心文件速查

- `minimind_src/model/`
  - `model_minimind.py`：MiniMind 架构实现（RMSNorm、RoPE、Attention/GQA、MoE FFN、forward、generate）
  - `model_lora.py`：LoRA 注入/保存/合并（`apply_lora`、`save_lora`、`merge_lora`）
- `minimind_src/dataset/`
  - `lm_dataset.py`：数据预处理与 labels/mask 生成（Pretrain/SFT/DPO/RL/AgentRL）
- `minimind_src/trainer/`
  - `trainer_utils.py`：初始化/分布式/seed/学习率/检查点保存与恢复
  - `rollout_engine.py`：RL rollout 与 per-token logprob 计算（Torch/SGLang 后端）
  - `train_pretrain.py` / `train_full_sft.py`：预训练与全量 SFT 训练循环
  - `train_lora.py`：LoRA 微调训练循环
  - `train_dpo.py`：DPO 偏好对齐
  - `train_distillation.py`：知识蒸馏（CE + KD）
  - `train_ppo.py`：PPO 强化学习
  - `train_grpo.py`：GRPO 强化学习
  - `train_agent.py`：AgentRL（多轮 tool_call + reward 规则 + completion_mask）
- `minimind_src/scripts/`（推理/服务/工具链）
  - `eval_llm.py`：本地加载权重并对话生成
  - `serve_openai_api.py`：OpenAI 兼容 SSE 流式 `/v1/chat/completions`，解析 `<think>/<tool_call>`
  - `chat_api.py`：OpenAI 兼容客户端示例
  - `web_demo.py`：Streamlit 网页端（流式渲染 + 工具回填重生成）
  - `eval_toolcall.py`：ToolCall 评测（解析工具标签 -> 执行 -> 回填 -> 继续生成）
  - `convert_model.py`：torch<->transformers 格式转换、chat_template jinja/json 互转
- `minimind_src/minimind-3/`（模型工件）
  - `chat_template.jinja`：对话模板协议（`<|im_start|>/<|im_end|>、<think>、<tool_call>、<tool_response>`）
  - `tokenizer_config.json`：特殊 token/词表与 chat_template 字段
  - `config.json`：结构参数与 RoPE 配置（如 `rope_theta`、heads、vocab_size）

---

## 学习笔记导航图（按建议顺序）

1. [00_项目总览与全局数据流](learning_notes/00_项目总览与全局数据流.md)
2. [01_核心模型 MiniMind（逐行+张量维度）](learning_notes/01_核心模型%20MiniMind（逐行+张量维度）.md)
3. [02_LoRA 适配与合并（逐行）](learning_notes/02_LoRA%20适配与合并（逐行）.md)
4. [03_数据集与对话模板（逐行+labels_mask 维度）](learning_notes/03_数据集与对话模板（逐行+labels_mask%20维度）.md)
5. [04_训练基础设施（分布式-检查点-采样器）](learning_notes/04_训练基础设施（分布式-检查点-采样器）.md)
6. [05_预训练与全量 SFT 训练循环](learning_notes/05_预训练与全量%20SFT%20训练循环.md)
7. [06_LoRA 微调训练](learning_notes/06_LoRA%20微调训练.md)
8. [07_DPO（偏好对齐）](learning_notes/07_DPO（偏好对齐）.md)
9. [08_知识蒸馏](learning_notes/08_知识蒸馏.md)
10. [09_Rollout 引擎与 per-token logprob 计算](learning_notes/09_Rollout%20引擎与%20per-token%20logprob%20计算.md)
11. [10_PPO 强化学习](learning_notes/10_PPO%20强化学习.md)
12. [11_GRPO 强化学习（group 相对策略）](learning_notes/11_GRPO%20强化学习（group%20相对策略）.md)
13. [12_AgentRL（ToolCall 多轮与奖励规则）](learning_notes/12_AgentRL（ToolCall%20多轮与奖励规则）.md)
14. [13_推理与服务脚本（加载-生成-流式-工具解析-模型转换）](learning_notes/13_推理与服务脚本（加载-生成-流式-工具解析-模型转换）.md)
15. [14_minimind-3/模型工件与 tokenizer_chat_template 解析（单独章）](learning_notes/14_minimind-3/模型工件与%20tokenizer_chat_template%20解析（单独章）.md)

