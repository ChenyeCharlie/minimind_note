# 02_LoRA 适配与合并（逐行）

本章逐行解析 `minimind_src/model/model_lora.py` 中的 LoRA 实现方式，并重点说明：

- 注入时 `Linear` 的 forward 是如何被“残差式叠加”到原输出里的
- LoRA 权重的 state_dict 键如何命名（影响加载/保存/合并）
- 维度关系：`x: [..., in_features] -> y: [..., out_features]`，以及 `(B @ A)` 的矩阵形状如何与原 `Linear.weight` 对齐

---

## 1. Import（逐行）

```python
1: import torch
2: from torch import optim, nn
```

- 第 1 行：用于加载/保存权重（`torch.load/torch.save`）以及 tensor 运算。
- 第 2 行：`nn` 用于定义模块；`optim` 在本文件里未被实际使用（训练脚本用）。

---

## 2. LoRA 子模块：`class LoRA(nn.Module)`

### 2.1 init：构建低秩分解（逐行+维度）

```python
6: class LoRA(nn.Module):
7:     def __init__(self, in_features, out_features, rank):
8:         super().__init__()
9:         self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
10:         self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
11:         self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
12:         # 矩阵A高斯初始化
13:         self.A.weight.data.normal_(mean=0.0, std=0.02)
14:         # 矩阵B全0初始化
15:         self.B.weight.data.zero_()
```

维度（按 `Linear` 的权重约定：`weight: [out_features, in_features]`）：

- `self.A = Linear(in_features -> rank)`
  - `A.weight: [rank, in_features]`
- `self.B = Linear(rank -> out_features)`
  - `B.weight: [out_features, rank]`
- LoRA 的目标是学习一个低秩增量：
  - `DeltaW = B @ A`
  - 其中 `DeltaW.shape = [out_features, in_features]`
  - 从而可以与原 `Linear.weight.shape=[out_features,in_features]` 相加（需要形状严格对齐）。

初始化动机：

- `A` 正态初始化：提供随机特征投影空间
- `B` 全 0 初始化：初始时 `DeltaW=0`，因此模型初始行为与未训练 LoRA 基线一致（更稳定的微调起点）。

### 2.2 forward：`B(A(x))`（逐行+维度）

```python
17:     def forward(self, x):
18:         return self.B(self.A(x))
```

维度追踪（假设输入最后一维是 `in_features`）：

- `x: [..., in_features]`
- `A(x)`: `[..., rank]`
- `B(A(x))`: `[..., out_features]`

这正是 LoRA 增量分支输出的 shape，与原 `Linear` 的输出可直接相加。

---

## 3. 注入 LoRA：`apply_lora(model, rank=16)`（逐行+关键机制）

### 3.1 遍历所有子模块并筛选 `nn.Linear`

```python
21: def apply_lora(model, rank=16):
22:     for name, module in model.named_modules():
23:         if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
24:             lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
25:             setattr(module, "lora", lora)
26:             original_forward = module.forward
```

逐行含义：

- 第 22 行：`named_modules()` 让你拿到每个子模块的“层级名称”（用于后续 state_dict 对齐）。
- 第 23 行：只对满足 `weight` 为**方阵**的 `Linear` 做 LoRA 注入：
  - `Linear.weight.shape = [out_features, in_features]`
  - 条件 `out_features == in_features` 才允许注入
- 第 24 行：构建 LoRA：
  - 传入 `in_features=module.weight.shape[0]` 与 `out_features=module.weight.shape[1]`
  - 由于上一步保证是方阵，这两个值相等，因此不会发生 `in/out` 搭错问题（若非方阵则直接跳过）。
- 第 25 行：把 LoRA 作为新属性挂到该 `Linear` 模块上：`module.lora = lora`
  - 这样 `model.parameters()` 会包含 LoRA 参数（训练脚本会按名字筛选 `lora` 参数并设置 `requires_grad=True`）。
- 第 26 行：保存原始 forward 以便后续“叠加式”调用。

### 3.2 显式绑定 forward：残差叠加（逐行+维度）

```python
28:             def forward_with_lora(x, layer1=original_forward, layer2=lora):
29:                 return layer1(x) + layer2(x)
30: 
31:             module.forward = forward_with_lora
```

关键点：

- 注入方式不是改计算图里显式新增一个并行分支模块，而是直接把 `Linear.forward` 替换成“原输出 + LoRA 输出”的残差形式。
- 维度要求：
  - `layer1(x)`: `[... , out_features]`
  - `layer2(x)`: `[... , out_features]`
  - 因此二者相加可成立。

动机：

- 这样微调时只需要训练 `module.lora` 参数，原 `Linear.weight` 默认保持冻结（由训练脚本控制 `requires_grad`）。

---

## 4. 加载 LoRA：`load_lora(model, path)`（逐行+state_dict 键规则）

```python
35: def load_lora(model, path):
36:     state_dict = torch.load(path, map_location=model.device)
37:     state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

39:     for name, module in model.named_modules():
40:         if hasattr(module, 'lora'):
41:             lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
42:             module.lora.load_state_dict(lora_state)
```

逐行含义与键对齐：

- 第 36 行：读取保存的 LoRA 权重文件（通常是 DDP 中保存的半精度 CPU tensor）。
- 第 37 行：剔除可能存在的 `module.` 前缀：
  - 这是为兼容 DDP 保存的 key（`DistributedDataParallel` 会在外面包一层 `module`）。
- 第 41 行的核心：
  - `state_dict` 里 key 的格式（由 `save_lora` 决定）大致是：`{layer_name}.lora.A.weight`、`{layer_name}.lora.B.weight`
  - 对于当前 `name` 对应的 `module`，筛出包含 `f'{name}.lora.'` 的条目
  - 再把前缀 `'{name}.lora.'` 去掉，剩下的 key 形如 `A.weight` / `B.weight`
  - 这样 `module.lora.load_state_dict(lora_state)` 才能匹配到 `LoRA` 内部子模块 `A`/`B` 的权重键。

---

## 5. 保存 LoRA：`save_lora(model, path)`（逐行+键规则）

```python
45: def save_lora(model, path):
46:     raw_model = getattr(model, '_orig_mod', model)
47:     state_dict = {}
48:     for name, module in raw_model.named_modules():
49:         if hasattr(module, 'lora'):
50:             clean_name = name[7:] if name.startswith("module.") else name
51:             lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
52:             state_dict.update(lora_state)
53:     torch.save(state_dict, path)
```

关键点：

- 第 46 行：`_orig_mod` 用于 DDP/torch.compile 场景的“解包”（避免 key 里多一层 wrapper）。
- 第 50 行：同样去掉 `module.` 前缀以提高可移植性。
- 第 51 行：保存时把 LoRA 内部权重 key 扩展为全路径：
  - `module.lora.state_dict()` 的 key 是 `A.weight`、`B.weight`
  - 保存时拼成：`{clean_name}.lora.A.weight`、`{clean_name}.lora.B.weight`
- `.cpu().half()`：保存成 fp16，减少文件体积（代价是精度略损）。

---

## 6. 合并 LoRA：`merge_lora(model, lora_path, save_path)`（逐行+矩阵乘法维度）

```python
56: def merge_lora(model, lora_path, save_path):
57:     load_lora(model, lora_path)
58:     raw_model = getattr(model, '_orig_mod', model)
59:     state_dict = {k: v.cpu().half() for k, v in raw_model.state_dict().items() if '.lora.' not in k}

60:     for name, module in raw_model.named_modules():
61:         if isinstance(module, nn.Linear) and '.lora.' not in name:
62:             state_dict[f'{name}.weight'] = module.weight.data.clone().cpu().half()
63:             if hasattr(module, 'lora'):
64:                 state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data).cpu().half()
65:     torch.save(state_dict, save_path)
```

逐步理解：

1. `load_lora(model, lora_path)`（第 57 行）
   - 让当前 `model` 里的每个注入点都有对应的 `module.lora.A/B` 参数。

2. 初始化要保存的 `state_dict`（第 59 行）
   - 把所有不包含 `.lora.` 的参数直接拷贝并转半精度保存
   - 因此输出文件将是“合并后的 dense 权重文件”，不再包含 LoRA 子模块参数。

3. 合并每个 `nn.Linear`（第 61-65 行）
   - 更新权重：
     - `state_dict[f'{name}.weight']` 先等于原 `module.weight`（半精度）
     - 若该 `Linear` 有 `module.lora`：
       - 计算 `B @ A`

维度校验（最重要）：

- 对于注入时 LoRA 只作用于“方阵 Linear”：
  - `module.weight: [out_features, in_features]` 且 `out_features == in_features = d`
- `module.lora.B.weight: [out_features, rank]` 即 `[d, r]`
- `module.lora.A.weight: [rank, in_features]` 即 `[r, d]`
- `B @ A`：`[d, r] @ [r, d] -> [d, d]`
- 与 `module.weight: [d, d]` 完全一致，因此可以做原地相加：
  - `W_merged = W + DeltaW`

动机：

- 合并后可以把模型当成“无 LoRA 分支的普通线性层”，直接加载更方便（例如部署或转换到其他框架）。

---

## 7. 与训练脚本的衔接（你后续必会用到）

虽然本章只讲 `model_lora.py`，但你后续读 `train_lora.py` 时需要把以下事实牢牢记住：

- `apply_lora(model)` 会把 LoRA 模块挂在每个目标 `nn.Linear` 上，并替换它的 `forward`
- `save_lora(model)` 只保存 LoRA 子模块权重（`{layer}.lora.A/B.weight`）
- `merge_lora(...)` 会把 LoRA 权重合并进 dense `Linear.weight` 并保存一个不含 `.lora.` 的 state_dict

因此：

- 训练阶段：通常用 `apply_lora` + `save_lora`
- 部署/转换阶段：可能用 `merge_lora` 生成“合并 dense 权重”

