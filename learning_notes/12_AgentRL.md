# 12_AgentRL（ToolCall 多轮与奖励规则）

本章对应 `minimind_src/trainer/train_agent.py`，实现的是针对 Agent 能力的强化学习：模型需要在生成过程中与环境（工具）进行多轮交互。

**核心挑战**：
1. **多轮序列打包**：如何将（模型生成 -> 工具执行 -> 模型生成）这种不连续的交互拼接成一个 Tensor 喂给训练？
2. **掩码隔离 (Masking)**：如何确保 Actor 只对模型自己吐出的 Token 负责，而不对“工具返回的结果”负责？
3. **复合奖励**：如何结合格式分、逻辑分和最终答案的 Ground Truth（GT）命中分？

---

## 1. 模拟环境：工具执行逻辑

```python
83: def execute_tool(name, args):
...
87:         signal.signal(signal.SIGALRM, ...) # 设置 1 秒硬超时
88:         signal.alarm(1)
89:         return fn(args)
```

- **动机**：为了让 RL 训练闭环，项目在本地 Mock 了 6 个常用工具（数学、汇率、翻译等）。
- **安全**：使用 `signal.alarm` 防止 `eval` 或复杂逻辑在训练采样过程中死锁挂起。

---

## 2. 核心采样：`rollout_single` 的掩码哲学

这是理解 AgentRL 的灵魂所在。函数生成的 `response_ids` 包含了多轮对话，但其对应的 `response_mask` 却不是全 1。

### 2.1 动作 Token (Model Action)
当模型正在思考或决定调用工具时：
- 生成 Token -> 存入 `response_ids`。
- **`response_mask` 置 1**。
- 意味着：Actor 模型的策略（Logprob）直接决定了这些 Token 的产生，需要通过 RL 进行强化。

### 2.2 观察 Token (Environment Observation)
当模型产生 `<tool_call>` 后，我们解析 JSON 并在 Python 中执行 `execute_tool`，得到 `{"result": "..."}`。
- 将结果转换为 Token -> 追加到 `response_ids`。
- **`response_mask` 置 0**。
- **动机**：这些 Token 是环境反馈给模型的“输入”，模型对其产生没有控制权。在计算 PPO/GRPO Loss 时，这些位置必须被屏蔽，否则 Actor 会试图去预测环境的输出，导致严重的训练偏移。

---

## 3. 数据打包逻辑：从 List 到 Tensor

在 `rl_train_epoch` 中，我们将多轮采样的结果进行对齐：

```python
255: ids = p + r # Prompt + Response(包含多轮交互)
256: mask = [0] * len(p) + m # m 就是上面的 response_mask
257: old_logps = [0.0] * max(len(p) - 1, 0) + old_lp
```

维度追踪：
- `input_ids`: `[N, max_total_len]` (补齐 Padding)。
- `completion_mask`: `[N, max_total_len-1]` (基于 `mask` 右移一位)。
- `old_per_token_logps`: `[N, max_total_len-1]`。

**对齐意义**：
- 在 `per_token_logps` 计算时，只有那些在交互中标记为 `1` 的动作 Token 才会计算 Ratio。
- 工具返回的 Token 虽然在 `input_ids` 中作为上下文，但在 Loss 计算中被完全忽略。

---

## 4. 复合奖励规则 (`calculate_rewards`)

AgentRL 的奖励函数（Reward Function）比基础模型复杂得多：

### 4.1 格式分 (Format)
- 检查 `<tool_call>` 是否闭合。
- 惩罚非法 JSON 格式。

### 4.2 逻辑分 (Process)
- 如果没有调用工具：使用奖励模型（Reward Model）打分。
- 如果调用了工具：计算模型调用的参数是否合法、工具名称是否正确。

### 4.3 终局分 (Outcome)
- **GT 验证**：使用正则和数值近似匹配，检查最终生成的答案中是否包含 `gt` 里的数值或字符串。
- `reward += 2.5 * (命中比例)`：这是最重要的奖励分，直接引导模型走向正确答案。

---

## 5. 本章总结

通过本章，你应该掌握 AgentRL 的三大支柱：
1. **多轮 Rollout**：利用 `TextIteratorStreamer` 和子线程实现的交互生成。
2. **选择性 Mask**：`mask=1` (模型动作) vs `mask=0` (环境反馈)。
3. **基于 GT 的 Reward**：不仅看过程（推理模型），更看结果（工具执行后的答案正确性）。

下一章 `13_推理与服务脚本` 将向你展示如何将这一整套复杂的“标签协议”暴露给前端 UI 或 OpenAI 客户端。
