# T2L-Style Meta-Network for π0.5 LoRA Generation

基于 [Text-to-LoRA](https://github.com/sakanaai/text-to-lora) (Sakana AI) 的元网络架构,用于根据条件证据生成 π0.5 VLA 策略的 LoRA 权重。

## 核心创新

相比 V4DirectABHyperNetwork (713M 参数),T2L-style 架构通过以下方式解决"通用 LoRA 塌缩"问题:

1. **Layer Embeddings**: 学习层深度和模块类型的嵌入,使网络能区分不同层
2. **共享主干**: Mixer + 2× ResidualBlocks 处理拼接的 [task_emb + depth_emb + type_emb]
3. **按模块类型输出头**: 每个形状组一个 Linear 头(17 组)
4. **重建锚点**: 预训练时回归到已有的专家 LoRA,强制输入依赖

## 架构组件

### 1. TwoBranchConditionEncoder

双分支条件编码器,融合语义和几何证据:

```python
from mlvla.meta.t2l_style import TwoBranchConditionEncoder

encoder = TwoBranchConditionEncoder(
    dino_dim=1024,      # DINO 特征维度
    geo_dim=7,          # 外参维度(view7)或 2048(vggt)
    hidden_dim=256,
    task_emb_size=256,
    fusion="additive",  # "additive" 或 "concat"
)

# 输入
semantic = torch.randn(B, K, 4, 1024)   # K 个证据片段,每个 4 帧
geometric = torch.randn(B, K, 7)        # K 个位姿

# 输出
task_emb = encoder(semantic, geometric)  # [B, 256]
```

**融合策略**:
- **Additive** (默认): LayerNorm + 加法,支持零样本组合(camera_L2 + lighting_L3 = 向量和)
- **Concat**: 拼接后接 MLP,更强的非线性但可能过拟合

### 2. T2LStyleHyperNet

生成 LoRA A/B 矩阵的超网络:

```python
from mlvla.meta.t2l_style import T2LStyleHyperNet

hypernet = T2LStyleHyperNet(
    module_shapes=module_shapes,  # {group_key: {"A": (r, in), "B": (out, r)}}
    max_layers=168,               # π0.5 层数
    task_emb_size=256,
    depth_emb_size=64,
    type_emb_size=64,
    trunk_hidden=512,
    head_in_size=512,
    lora_rank=16,
    shared_AB_head=False,         # True: 40M 参数变体
)

# 生成 LoRA
A, B = hypernet(task_emb, layer_indices, module_key)
# A: [B, r, in_features]
# B: [B, out_features, r]
```

**参数量**:
- 完整模型 (~17 组): ~72M (vs V4 713M, 10× 减少)
- Shared head 变体: ~10M (70× 减少)

### 3. ReconstructionLoss

重建损失,强制生成的 LoRA 匹配专家 LoRA:

```python
from mlvla.meta.t2l_style import ReconstructionLoss

loss_fn = ReconstructionLoss(pred_z_score=False)

loss_dict = loss_fn(
    pred_A, pred_B,        # 生成的 LoRA
    target_A, target_B,    # 专家 LoRA
)
# loss_dict["loss"]: MSE 损失
# loss_dict["unnorm_err"]: 未归一化的 L1 误差
```

## 训练流程

### Phase 1: 重建预训练

防止塌缩的关键步骤:回归到 12 个 oracle 专家 LoRA。

```python
from mlvla.meta.t2l_style import ReconstructionTrainer

trainer = ReconstructionTrainer(
    hypernet=hypernet,
    condition_encoder=encoder,
    oracle_loras=oracle_loras,  # {domain: {module_key: {"A": [L, r, in], "B": [L, out, r]}}}
    evidence_bank=evidence_bank,
    device=device,
    lr=1e-3,
)

trainer.train(
    n_epochs=100,
    batch_size=8,
    n_batches_per_epoch=100,
)
```

**成功标准**:
- 验证集余弦相似度 > 0.95
- 跨域相对 L2 距离 >> 0.001 (证据真正起作用)

### Phase 2: 端到端训练

在 `train_bridge.py` 中集成,联合优化动作损失和重建损失:

```python
# 伪代码
task_emb = condition_encoder(dino_features, poses)
A, B = hypernet(task_emb, layer_indices, module_key)

# 注入 LoRA 到 π0.5
inject_lora(pi05_model, A, B)

# 计算动作损失(通过 JAX 后端)
action_loss = compute_action_loss(pi05_model, actions)

# 计算重建损失(辅助项)
recon_loss = compute_recon_loss(A, B, oracle_loras)

# 总损失
total_loss = action_loss + lambda_recon * recon_loss
```

**超参数**:
- `lambda_recon`: 0.1 (重建损失权重)
- `lr`: 8.5e-4
- 训练步数: 30k

## 与 V4 的对比

| 组件 | V4 (当前) | T2L-Style |
|------|----------|-----------|
| **条件编码** | Concat[DINO, pose] → MLP | 双分支投影 → 加法融合 |
| **层身份** | 无 | 深度嵌入 + 类型嵌入 |
| **主干** | ResidualBlock | Mixer + 2× ResBlock |
| **输出头** | 17 形状组头 | 17 形状组头 |
| **参数** | 713M | 280M (或 40M) |
| **训练** | 仅 E2E | 重建锚点 + E2E |
| **塌缩?** | 是 (rel-L2 ~0.0001) | 目标: rel-L2 > 0.1 |

## 文件结构

```
src/mlvla/meta/t2l_style/
├── __init__.py              # 导出所有类
├── condition_encoder.py     # TwoBranchConditionEncoder
├── hypernet.py              # T2LStyleHyperNet
├── losses.py                # ReconstructionLoss, CombinedLoss
├── train_recon.py           # 重建预训练器
└── README.md                # 本文档
```

## 下一步

1. **集成到 train_bridge.py**: 添加 T2L 变体分发
2. **准备 oracle LoRA**: 从 12 个域专家加载 LoRA
3. **重建预训练**: 运行 100 epochs,验证余弦 > 0.95
4. **E2E 训练**: 30k 步,评估 12 域闭环性能
5. **组合泛化测试**: camera+lighting, noise+texture 等

## 参考文献

- [Text-to-LoRA: Large Language Models as Implicit Task Adapters](https://arxiv.org/abs/2311.10473)
- ML-VLA V4 实现: `src/mlvla/meta/hypernet/v4_head.py`
