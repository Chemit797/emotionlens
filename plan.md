# 五组实验执行计划

> **目标**: 补全因果链条，让报告从"老师建议的 FER2013+VGG"到"我们的 EmotionLens 系统"逻辑完整。
> **原则**: 控制变量法——每次只变一个因素，隔离架构/数据/预训练各自的影响。
> **统一超参**: epochs=35, lr=3e-4, wd=1e-4, bs=128, AdamW, CosineAnnealing, AMP

---

## 实验总览

```
实验1: VGG16 + FER2013 (hard labels, 7-class)     → 对接老师 baseline
实验2: ResNet18 + FERPlus (soft labels, 8-class)   → 同数据不同架构（对比 EfficientNet）
实验3: 数据消融对比                                  → FERPlus-only vs FERPlus+RAFDB+Self
实验4: 跨域统一评测表                                → 所有模型 × 所有测试集
实验5: FER 库 baseline                              → 第三方参照
```

| 实验 | 需训练 | 需推理 | 预计时间(GPU) | 产出 |
|------|--------|--------|---------------|------|
| 1 | ✅ VGG16 | ✅ 2个测试集 | ~2h | checkpoint + metrics + 图表 |
| 2 | ✅ ResNet18 | ✅ 2个测试集 | ~1.5h | checkpoint + metrics + 图表 |
| 3 | ❌ | ✅ 对比评测 | ~15min | 消融对比表 + 图表 |
| 4 | ❌ | ✅ 统一评测 | ~30min | 跨域大表 + 雷达图 |
| 5 | ❌ | ✅ FER库评测 | ~20min | FER库 baseline 指标 |

---

## 统一超参约定

所有训练脚本保持一致（除非架构本身强制不同）：

| 参数 | 值 | 备注 |
|------|-----|------|
| epochs | 35 | 与现有 EfficientNet 一致 |
| lr | 3e-4 | AdamW 初始学习率 |
| wd | 1e-4 | 权重衰减 |
| batch_size | 128 | |
| optimizer | AdamW | |
| scheduler | CosineAnnealingLR(opt, epochs) | 余弦退火到 0 |
| AMP | ✅ (CUDA only) | GradScaler + autocast |
| class balance | WeightedRandomSampler | 1.0/count per class |
| save strategy | best val_acc | 保存在 runs/{arch}_{timestamp}/best.pt |

| 架构特有 | VGG16 | ResNet18 | EfficientNet-B0 |
|----------|-------|----------|-----------------|
| img_size | 224 | 224 | 128 |
| in_chans | 3 | 3 (repeat gray→RGB) | 1 (grayscale) |
| pretrained | ImageNet | ImageNet | ImageNet |
| num_classes | 7 | 8 | 8 |
| loss | CrossEntropyLoss (hard) | soft_ce (soft labels) | soft_ce (soft labels) |

> **为什么不统一 img_size？** 每个架构的设计分辨率不同。VGG/ResNet 是 224×224，EfficientNet-B0 官方也是 224，但我们之前用 128 训好了 EfficientNet，没必要重训。只要在报告里标注清楚即可。跨架构对比时，img_size 是架构特征的一部分。

---

## 实验 1: VGG16 Baseline on FER2013

### 目的
- 复现老师建议的 "FER2013 + VGG16" 方案，作为报告的性能下限参照
- 证明我们从 VGG→ResNet→EfficientNet 的演进是有效的

### 数据
- `data/fer2013.csv` — 原始 FER2013，48×48 灰度，7 类硬标签
- 按 Usage 列拆分: Training / PublicTest(val) / PrivateTest(test)
- 不做软标签，直接用 CrossEntropyLoss

### 脚本
创建 `exp1_vgg_baseline.py`:
1. 读取 fer2013.csv，按 Usage 列拆 train/val/test
2. 数据增强（简化版，无 degrade，因为 FER2013 已经是 48×48）:
   - 水平翻转 (50%)
   - ±12° 旋转 (50%)
   - 亮度/对比度 jitter (50%)
   - Resize 到 224×224
   - 灰度 → 3 通道重复（适配 VGG pretrained）
   - ImageNet 归一化: mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
3. 模型: `torchvision.models.vgg16(weights="IMAGENET1K_V1")`
   - 替换最后一层: `classifier[6] = nn.Linear(4096, 7)`
4. 训练: epochs=35, lr=3e-4, AdamW, CosineAnnealing
5. 每 epoch 在 PublicTest 上验证，保存最佳 checkpoint
6. 最终在 PrivateTest 上测试
7. 额外: 在 self data 上评测（用现有 eval 逻辑）

### 产出
- `runs/vgg16_fer2013_YYYYMMDD_HHMMSS/best.pt`
- `runs/vgg16_fer2013_YYYYMMDD_HHMMSS/metrics.json` (含 val_best, test_acc, test_f1, history)
- `runs/vgg16_fer2013_YYYYMMDD_HHMMSS/confusion_matrix.png`

### 验证
- [ ] 训练完成，val_acc > 60% 即可（VGG16 on FER2013 典型 ~65-70%）
- [ ] test_acc 记录在 metrics.json
- [ ] Self data 评测结果

---

## 实验 2: ResNet18 on FERPlus（控制变量）

### 目的
- 用与 EfficientNet-B0 完全相同的数据 (FERPlus 8-class soft labels) 训练 ResNet18
- 隔离架构差异：数据相同 → 性能差异来自架构

### 数据
- `data/ferplus.parquet` — 与现有 EfficientNet 完全相同的训练/验证/测试集

### 关键处理
- FERPlus 图像是 48×48 灰度 → ResNet18 需要 224×224 RGB
- 方案: 在 dataset 中将灰度 repeat 3 次 → resize 到 224 → ImageNet 归一化
- 使用 pretrained ImageNet 权重

### 脚本
创建 `exp2_resnet_ferplus.py`:
1. 复用 `dataset.py` 的 FERPlusDataset 逻辑，但输出改为:
   - 灰度 repeat 为 3 通道
   - resize 到 224×224（而非 128）
   - ImageNet 归一化（而非 [-1,1]）
2. 模型: `timm.create_model("resnet18", pretrained=True, num_classes=8, in_chans=3)`
3. 训练参数同实验 1 / train.py

### 产出
- `runs/resnet18_ferplus_YYYYMMDD_HHMMSS/best.pt`
- `runs/resnet18_ferplus_YYYYMMDD_HHMMSS/metrics.json`
- `runs/resnet18_ferplus_YYYYMMDD_HHMMSS/confusion_matrix.png`

### 验证
- [ ] 训练完成，test_acc 记录
- [ ] 与 EfficientNet 对比: 这是公平的架构对比（同数据同设置）

---

## 实验 3: 数据消融 — 多源数据到底加了多少分

### 目的
- 回答 "RAFDB + 自采数据到底值不值得"
- 对比: ResNet18(FERPlus only) vs Classmate ResNet18(FERPlus+RAFDB+Self)

### 无需新训练
- ResNet18(FERPlus-only): 使用实验 2 训出的模型
- Classmate(FERPlus+RAFDB+Self): 使用现有 `runs/classmate_model/best.pt`

### 脚本
创建 `exp3_ablation.py`:
1. 加载两个 ResNet18 checkpoint
2. 在两个测试集上分别评测:
   - FERPlus test set
   - Self-collected data
3. 生成对比表和图表:
   - 同测试集上两个模型的 acc/f1 对比
   - Per-class recall 对比（重点关注 contempt 是否被同学模型学到）
   - Domain gap 对比: FERPlus→Self 的下降幅度

### 产出
- `runs/ablation/ablation_comparison.json`
- `runs/ablation/ablation_table.png` (对比柱状图)
- `runs/ablation/per_class_delta.png` (每类提升/下降)

### 验证
- [ ] 看懂 "加 RAFDB+Self 后，在 Self test 上提升 X%，但在 FERPlus test 上不一定有提升"
- [ ] 实验 2 的 ResNet18 在 self data 上的表现 vs classmate 在 self data 上的表现

---

## 实验 4: 跨域统一评测表

### 目的
- 一张表看完所有模型的跨域泛化能力
- 报告的"总成绩单"

### 评测矩阵

```
模型                          FERPlus test    Self data    FER2013 test
─────────────────────────────────────────────────────────────────────
VGG16 (FER2013)                    ?              ?             ?       ← Exp1
ResNet18 (FERPlus only)            ?              ?             ?       ← Exp2
ResNet18 (F+R+S, classmate)        ?            72.3%           ?       ← 已有部分
EfficientNet-B0 (FERPlus)        81.6%          44.8%           ?       ← 已有部分
FER 库                            —              ?              —       ← Exp5
```

### 脚本
创建 `exp4_cross_eval.py`:
1. 加载所有可用 checkpoint
2. 对每个 (model, test_set) 组合跑推理
3. 输出统一表格 (CSV + 终端打印)
4. 生成:
   - 雷达图: 每类 recall × 每个模型
   - Domain gap 柱状图: FERPlus → Self 的 acc 下降

### 关键处理
- 8-class 模型在 7-class 数据上评测时，去掉 contempt 维度
- 7-class 模型在 8-class(FERPlus) 数据上评测时，只比较 7 个共享类别
- VGG16 的 FER2013 test 就是它自己的测试集，不需要交叉

### 产出
- `runs/cross_eval/cross_eval_table.csv`
- `runs/cross_eval/cross_eval_radar.png`
- `runs/cross_eval/domain_gap_summary.png`

### 验证
- [ ] 表格所有格子都有数值（无遗漏）
- [ ] 趋势合理: Self data 上 F+R+S > F-only；FERPlus test 上两者接近

---

## 实验 5: FER 库 Baseline

### 目的
- 对比"调库派"方案，凸显自训练的价值
- FER 库也是 FER2013 训的 → 应该和 VGG16 baseline 表现类似

### 脚本
已有 `eval_fer.py`，直接运行即可。

### 产出
- FER 库在 self data 上的 acc + confusion matrix（终端输出，手工记录到汇总表）

### 验证
- [ ] eval_fer.py 运行完成（约 15-20 分钟，MTCNN 检测较慢）
- [ ] 结果记录到实验汇总

---

## 执行顺序

```
Phase 1 (并行): 实验 1 训练 ← 可以同时启动
               实验 2 训练 ← 可以同时启动

Phase 2:       实验 5 FER库 ← 独立运行，Phase 1 期间也可做

Phase 3 (依赖 Phase 1+2):
               实验 3 消融 ← 需要 Exp2 的 checkpoint
               实验 4 跨域 ← 需要 Exp1+Exp2 的 checkpoint
```

## 最终汇总 (所有实验完成后)

整合所有结果到一份 `EXPERIMENT_SUMMARY.md`:

```
1. Baseline (VGG16+ FER2013): test_acc=X%, self_acc=Y%
2. 架构升级 (FERPlus + EfficientNet): test_acc=81.6%, self_acc=44.8%
3. 数据升级 (FERPlus+RAFDB+Self + ResNet18): test_acc=70%+, self_acc=72.3%
4. 控制变量证明:
   - 同数据下 EfficientNet vs ResNet18 → 架构提升 = X%
   - 同架构下 FERPlus-only vs F+R+S → 数据提升 = X% (尤其在 self data 上)
5. FER库参照: self_acc=Y%
6. 结论: 数据多样性对真实场景泛化的提升 >> 架构复杂度
```

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `exp1_vgg_baseline.py` | VGG16 训练脚本 |
| `exp2_resnet_ferplus.py` | ResNet18 训练脚本 |
| `exp3_ablation.py` | 消融对比 |
| `exp4_cross_eval.py` | 跨域统一评测 |
| `eval_fer.py` (已有) | FER库 baseline |
| `runs/exp1_vgg16_*/` | VGG16 产出 |
| `runs/exp2_resnet18_*/` | ResNet18 产出 |
| `runs/ablation/` | 消融产出 |
| `runs/cross_eval/` | 跨域评测产出 |
| `EXPERIMENT_SUMMARY.md` | 最终汇总 |
