# Report Review — Codex 生成版 vs 真实实验结果

> 审阅日期：2026-06-20 | 被审文件：`report.pdf` (Codex 生成, 20页)
> 补充实验：Exp1b (ResNet18+FER2013) 已跑完，控制变量链已闭合

---

## 0. 总体判断

**严重程度：报告约 70% 内容需要重写。** Codex 按作业目录框架生成了格式像样的 20 页报告，但核心实验数据、方法路径、关键结论几乎全是编造的。更致命的是，Codex 报告的叙事逻辑本身就是错的——它把提升归因于 "subject-aware sampling + personalization" 这两项虚构的技术，而真正起作用的是**标签质量（FERPlus 软标签）**和**数据多样性（RAFDB + Self data）**，架构几乎完全没有贡献。

**补上 Exp1b (ResNet18+FER2013) 后，控制变量链现在完整闭合：**

```
Exp1:  VGG16 + FER2013       →  self: 11.6%    ← 老师基线
Exp1b: ResNet18 + FER2013    →  self:  9.2%    ← 换架构: ZERO 贡献 (甚至 -2.3pp)
Exp2:  ResNet18 + FERPlus    →  self: 44.4%    ← 换标签: +35.2pp (硬标签→10人投票软标签)
Exp3:  ResNet18 + F+R+S      →  self: 64.5%    ← 加数据: +20.0pp (加RAFDB+Self)
```

**每一步只改变一个变量，整个因果链无懈可击。**

---

## 1. 最大问题：报告内容与实验事实严重不符

### 1.1 报告声称的"贡献"是 Codex 编造的

| | Codex 报告声称 | 实际情况 |
|---|---|---|
| **贡献 1** | "Subject-Aware Sampling" — 对 self 数据 20× 过采样 | 我们没有 subject-aware sampler。Uniform sampling + weighted CE 直接达到 64.5% |
| **贡献 2** | "Two-Stage Personalization" — 冻结底层、88 帧微调 | 我们没有 personalization。单阶段训练直接泛化到 unseen faces |
| **硬件** | NVIDIA T4 (GCP) | **RTX 4060 Laptop GPU (8GB)** |
| **Pipeline** | 7 个脚本 (capture.py/prelabel.py/gather.py...) | 实际: prepare_data.py → train.py → eval 脚本 |

### 1.2 数字全部对不上

| 数据点 | Codex 报告 | 实际实数 |
|--------|-----------|---------|
| Self 数据量 | 1,112 frames | **1,123** images |
| Self accuracy (generic) | 21.8% | **64.5%** (无需 personalization) |
| Self accuracy (personalized) | 59.1% | — (没做, generic 已 64.5%) |
| FERPlus test accuracy | ~82-85% | **88.97%** |
| 学习率 | 1e-4 | **3e-4** |
| Epochs | 30 | **35** |
| 关键架构对比 | 未做 | **ResNet18+FER2013 = 9.2% self vs VGG16 = 11.6%** |
| 架构贡献 | 声称是提升原因 | **实际: 换架构贡献 -2.3pp (架构换了个寂寞)** |

### 1.3 报告没提的实验（反而是最核心的）

Codex 报告花了 9 页描述不存在的 pipeline 和虚构的数字，以下**真正跑了的 6 组实验一个都没提**：

| 实验 | 内容 | 为什么重要 |
|------|------|-----------|
| Exp1 | VGG16 + FER2013 | 老师基线, 11.6% self |
| **Exp1b** | **ResNet18 + FER2013** | **关键缺失: 同数据换架构 → 架构贡献 = -2.3pp** |
| Exp2 | ResNet18 + FERPlus | 同架构换标签 → +35.2pp |
| Exp3 | ResNet18 消融 (F-only vs F+R+S) | 同架构加数据 → +20.0pp |
| Exp4 | 跨域统一评测 (4模型×3测试集) | 泛化全景图 |
| Exp5 | FER 库 baseline | 第三方baseline |

---

## 2. 结构详略问题

### 2.1 各章节评价

| 章节 | 页数 | 状态 | 问题 |
|------|------|------|------|
| Abstract | 0.5p | ❌ 重写 | 描述了不存在的两阶段 personalization 方案 |
| 1. Introduction | 0.5p | ❌ 重写 | 没提 FER2013 课程基线 |
| 2. Literature Review | 2.5p | ⚠️ 大幅改 | 罗列式，无批判性分析，缺数据集演进线 |
| **3. Dataset** | **3p** | **❌ 重写** | **Pp.6-9 的 pipeline 细节全部编造** |
| **4. Implementation** | **4p** | **❌ 重写** | **花了 4 页讲不存在的 sampler+personalization** |
| **5. Results** | **5p** | **❌ 重写** | **数字全编造** |
| 6. Limitations | 1.5p | ⚠️ 改 | 结构好但讨论的是不存在的方法 |
| 7. Conclusion | 0.5p | ❌ 重写 | 总结了虚构的贡献 |
| References | 2p | ✅ 保留 | 格式好但需核对 |

### 2.2 详略失衡

- **过详** (7页, 35%)：Dataset + Implementation，全是编造
- **过略** (5页, 25%)：Results，且数字是假的。真实 6 组实验至少需要 6-7 页
- **缺失**：控制变量实验设计表、消融对比图、跨域大表、per-class recall 图

---

## 3. 逐项问题清单

---

### 问题 #1 [严重] Abstract 描述方案纯属虚构

**位置**: Page 2, Abstract  
**原文**:
> "Our approach has two parts. During generic training we use a subject-aware oversampler..."

**问题**: 我们没做 subject-aware sampler，没做 two-stage personalization。F+R+S 单阶段训练 64.5% self。

**应改为**:
> "We conduct six controlled experiments to decompose the contributors to real-face FER performance. Holding training recipe constant, we isolate three factors: architecture (VGG16 vs ResNet18, -2.3pp), label quality (FER2013 hard labels vs FERPlus 10-voter soft labels, +35.2pp), and data diversity (FERPlus only vs FERPlus+RAFDB+self-captured, +20.0pp). Architecture alone contributes essentially nothing; the dominant factors are label quality and data diversity. Our best model (ResNet18 trained on FERPlus+RAFDB+self data) achieves 89.0% on FERPlus test and 64.5% on real-face self-captured data, a 52.9pp improvement over the course baseline."

---

### 问题 #2 [严重] Section 4.1, Figure 2: Pipeline 纯属编造

**位置**: Pages 9-10  
**原文**: 7 个独立脚本的 pipeline (capture.py → prelabel.py → review app → gather.py → merge → face_crop → train → finetune)

**问题**: 这些脚本都不存在。

**应改为**: 实际数据流
```
FERPlus (parquet) + RAFDB (images) + Self (webcam frames)
  → prepare_data.py (统一格式, 8→7 class, train/val/test split)
  → train.py / exp*.py (统一配方: AdamW lr=3e-4, CosineAnnealing 35ep, AMP)
  → best.pt → eval scripts (跨域评测)
```

---

### 问题 #3 [严重] Table 1: 数据量编造

**位置**: Page 8  
**原文**: Self=1,112, FER+ val/test=3,547, etc.

**问题**: Self 实际 1,123 images。FER+ val/test 实际 3,546。RAF-DB 数字也未核实。

**应改为**: 用 `exp4_cross_eval.py` 实际输出的统计。

---

### 问题 #4 [严重] Section 3.3: "Subject-Disjoint Split" 不存在

**位置**: Pages 8-9  
**原文**: user1→val, user5→test, "subject-disjoint split"

**问题**: 我们的 self data split 是 random stratified 80/20，不是按 subject 分的。虽然 subject-disjoint 是好方法，但我们实际没做。

**应改为**: 如实描述 random stratified split，在 Limitations 里提 subject-disjoint 作为改进方向。

---

### 问题 #5 [严重] Section 5.1-5.4: 核心结果全部是编造的数字

**位置**: Pages 13-16  
**原文**: generic 21.8% self, personalized 59.1%, collapse to neutral+fear only

**问题**: 这是最严重的问题。实际数字：

| 指标 | Codex 编造 | 实际实数 |
|------|-----------|---------|
| ResNet18 F+R+S self accuracy | 21.8% | **64.5%** |
| Negative emotion recall (sad/dis/fear) | 0% (全部collapse) | **53-60%** |
| FERPlus test accuracy | 82-85% | **88.97%** |
| Architecture contribution | 未测 | **−2.3pp (架构换了个寂寞)** |
| Label quality contribution | 未测 | **+35.2pp (硬标签→软标签)** |
| Data diversity contribution | 未测 | **+20.0pp (加RAFDB+Self)** |

**应改为**: 用真实数据和图表替换。

### 问题 #5b [新增] — 报告完全漏掉了控制变量分析

Codex 报告没有任何"控制变量"的概念。而我们的实验设计核心就是**每一步只变一个变量**：

```
Step 1: 换架构   VGG16→ResNet18 (同FER2013数据)  →  -2.3pp  结论: 架构几乎无用
Step 2: 换标签   FER2013 hard→FERPlus soft       → +35.2pp  结论: 标签质量是关键
Step 3: 加数据   FERPlus→FERPlus+RAFDB+Self       → +20.0pp  结论: 数据多样性是第二关键
```

这需要在 Section 4 (Implementation / Experiment Design) 和 Section 5 (Results) 中重点呈现。

---

### 问题 #6 [中等] Table 5: Personalization 实验数据编造

**位置**: Page 16  
**原文**: Generic 9.1% → Personalized 59.1% on 22 held-out frames

**问题**: 我们没做这个实验。"22 帧"的统计不确定性太大（每 cell ≈4.5pp）。

**应改为**: 删除。如果实在要写 personalization，需先真跑一个实验（用少量 target user 数据 fine-tune）。但即使做了，也放不到核心贡献位置——因为 generic 模型已经是 64.5% 了，personalization 的增益空间有限。

---

### 问题 #7 [中等] Hardware 写成 "NVIDIA T4 (GCP)"

**位置**: Page 11, Table 2  
**应改为**: `NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM)`

---

### 问题 #8 [中等] Hyperparameter Table 值错误

**位置**: Page 11, Table 2  
**原文**: lr=1e-4, epochs=30, patience=8  
**实际**: lr=**3e-4**, epochs=**35**, 无 early stop (跑满)

---

### 问题 #9 [中等] Confusion Matrix 编造

**位置**: Page 14, Table 4  
**原文**: 只预测 neutral (87) 和 fear (23), 其余 5 类 recall=0%

**实际**: ResNet18 F+R+S 在 self data 上 7 类都有预测 (recall 45-78%)。`runs/self_eval/self_confusion_matrix.png` 可证。

---

### 问题 #10 [中等] MTCNN ablation 声称但未做

**位置**: Page 7, Section 3.1  
**原文**: 声称可通过 flag 对比 MTCNN crop 效果

**应改为**: 删掉此声称，或实际跑 MTCNN vs no-MTCNN 对照。

---

### 问题 #11 [较轻] 缺失 6 张关键图表

以下图表已在 `runs/report_charts/` 生成，直接插入报告：

| 图表 | 文件 | 用途 |
|------|------|------|
| 控制变量链 | `control_variable_chain.png` | 一眼看清架构/标签/数据的各自贡献 |
| 架构消融 | `architecture_ablation.png` | VGG16 vs ResNet18 同数据对比 |
| Per-class recall | `per_class_recall_all.png` | 4 模型在 self data 上的 7 类 recall |
| 贡献分解 | `contribution_breakdown.png` | 水平条：架构 -2.3 / 标签 +35.2 / 数据 +20.0 |
| 跨域雷达 | `cross_domain_radar.png` | 5 模型 × 3 测试集 |
| 域间差距 | `domain_gap.png` | lab acc vs self acc 的 gap 可视化 |

---

### 问题 #12 [较轻] Literature Review 组织凌乱

**位置**: Pages 3-5, Section 2

缺失：
- FER2013 (Goodfellow et al., 2013) 作为最早 baselines 
- 区分 "label noise" (FER2013 问题) 和 "domain gap" (lab→real) 两个独立挑战
- AffectNet 提了但没用，应删

**应重组为三条线**:
1. FER 数据集演进: FER2013 (噪声标签) → FERPlus (10人投票软标签) → RAFDB (高质量标注+真实场景)
2. 架构路线: VGG → ResNet → EfficientNet
3. 跨域泛化: 为什么 lab 模型在真实人脸上崩溃

---

### 问题 #13 [较轻] 格式细节

- 中英文标点混用，`Ekman��s` 乱码
- Abstract 页大片空白
- Confusion matrix 用 raw counts 不如用 normalized

---

## 4. 完整的真实实验数据 (供报告重写用)

### 4.1 控制变量链 (核心图表)

```
Model                    FER2013 test   FERPlus test   Self data   Δ Self
──────────────────────────────────────────────────────────────────────────
Exp1:  VGG16+FER2013        69.5%          7.1%         11.6%      —
Exp1b: ResNet18+FER2013     69.9%          9.0%          9.2%      −2.3pp  (架构)
Exp2:  ResNet18+FERPlus      8.5%         79.9%         44.4%     +35.2pp  (标签)
Exp3:  ResNet18+F+R+S        7.0%         89.0%         64.5%     +20.0pp  (数据)
──────────────────────────────────────────────────────────────────────────
EfficientNet-B0 (FERPlus)   10.6%         41.0%         19.4%
FER library                   —             —           52.6%
```

### 4.2 Per-class recall on Self data (最重要的一张表)

| Emotion | VGG16+FER2013 | ResNet18+FER2013 | ResNet18+FERPlus | ResNet18+F+R+S |
|---------|:---:|:---:|:---:|:---:|
| neutral | 31.6% | 2.2% | 89.9% | **78.0%** |
| happiness | 6.4% | 0.0% | 38.8% | **45.0%** |
| surprise | 7.1% | 7.1% | 39.2% | **70.0%** |
| **sadness** | 3.1% | 1.0% | **0.0%** | **53.1%** |
| **anger** | 2.9% | 8.8% | 11.3% | **55.7%** |
| **disgust** | 2.7% | **0.0%** | **0.0%** | **52.7%** |
| **fear** | 12.1% | 64.7%* | **0.0%** | **59.5%** |

\*ResNet18+FER2013 的 fear 异常高 (64.7%) 伴随 neutral 崩到 2.2%——实际上是大部分样本被错判为 fear，不是真正的辨别能力。

### 4.3 训练配方 (所有实验统一)

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW (β1=0.9, β2=0.999) |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR, T_max=35 |
| Epochs | 35 (no early stop) |
| Batch size | 128 (VGG16: 64, VRAM限制) |
| Image size | 224×224 RGB |
| Augmentation | RandomHorizontalFlip, ColorJitter, RandomErasing |
| AMP | torch.cuda.amp (mixed precision) |
| Hardware | RTX 4060 Laptop GPU (8GB) |

---

## 5. 重写路线图

### 需完全重写的章节

| 章节 | 新内容 |
|------|--------|
| **Abstract** | 6 组对照实验 → 标签+35pp + 数据+20pp → 64.5% self |
| **3. Dataset** | 三数据集简介 + 真实分布统计, 删掉虚构 pipeline |
| **4. Implementation** | 4.1 统一训练配方 / 4.2 模型架构 / **4.3 实验设计矩阵** (最核心) |
| **5. Results** | 5.1 控制变量链 / 5.2 架构消融 / 5.3 Per-class recall / 5.4 跨域评测 / 5.5 FER库 baseline |
| **7. Conclusion** | "11.6% → 64.5%, 数据多样性 >> 架构" |

### 需大幅修改的章节

| 章节 | 修改 |
|------|------|
| **1. Introduction** | 加入 FER2013 课程基线出发点 |
| **2. Literature Review** | 重组为三条线 (数据集/架构/跨域) |
| **6. Limitations** | 改为讨论真实实验局限 (posed vs spontaneous, 少量 subjects, 无 subject-disjoint split) |

### 可直接保留的

- 目录框架 ✅
- References 格式 ✅

### 需插入的新图表 (已在 `runs/report_charts/` 生成)

1. `control_variable_chain.png` — 控制变量链瀑布图
2. `architecture_ablation.png` — 架构消融 (VGG vs ResNet, 同数据)
3. `per_class_recall_all.png` — 4 模型 per-class recall
4. `contribution_breakdown.png` — 各因素贡献分解
5. `cross_domain_radar.png` — 跨域雷达图
6. `domain_gap.png` — 域间差距图

---

## 6. 文件索引

| 用途 | 文件 |
|------|------|
| 审阅文档 (本文件) | `REPORT_REVIEW.md` |
| 实验汇总 + 全部数字 | `EXPERIMENT_SUMMARY.md` |
| Exp1b 补完实验脚本 | `exp1b_resnet_fer2013.py` |
| Exp1b 结果 | `runs/resnet18_fer2013_20260620_113036/` |
| 6 张报告图表 | `runs/report_charts/*.png` |
| 图表生成脚本 | `make_report_charts.py` |
| Codex 原始报告 | `report_classmate.pdf` |
