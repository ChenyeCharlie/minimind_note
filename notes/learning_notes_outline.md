# minimind 学习笔记大纲（待逐章展开）

## 章节列表
1. `learning_notes/00_项目总览与全局数据流.md`：项目总览与训练/推理端到端数据流
2. `learning_notes/01_核心模型 MiniMind（逐行+张量维度）.md`：`model_minimind.py` 全量解析
3. `learning_notes/02_LoRA 适配与合并（逐行）.md`：`model_lora.py` 全量解析
4. `learning_notes/03_数据集与对话模板（逐行+labels_mask 维度）.md`：`lm_dataset.py` 全量解析
5. `learning_notes/04_训练基础设施（分布式-检查点-采样器）.md`：`trainer_utils.py` 全量解析
6. `learning_notes/05_预训练与全量 SFT 训练循环.md`：`train_pretrain.py` + `train_full_sft.py`
7. `learning_notes/06_LoRA 微调训练.md`：`train_lora.py`
8. `learning_notes/07_DPO（偏好对齐）.md`：`train_dpo.py`
9. `learning_notes/08_知识蒸馏.md`：`train_distillation.py`
10. `learning_notes/09_Rollout 引擎与 per-token logprob 计算.md`：`rollout_engine.py`
11. `learning_notes/10_PPO 强化学习.md`：`train_ppo.py`
12. `learning_notes/11_GRPO 强化学习（group 相对策略）.md`：`train_grpo.py`
13. `learning_notes/12_AgentRL（ToolCall 多轮与奖励规则）.md`：`train_agent.py`
14. `learning_notes/13_推理与服务脚本（加载-生成-流式-工具解析-模型转换）.md`：`eval_llm.py` + `scripts/*`
15. `learning_notes/14_minimind-3/模型工件与 tokenizer_chat_template 解析（单独章）.md`：`minimind-3/` 工件相关内容

## 拆分选择
- `Rollout` 与 `per-token logprob` 放在 `09`。
- `PPO/GRPO/AgentRL` 的 loss/advantage/reward 与 mask 逻辑放在 `10/11/12`（实现贴近式讲解）。

