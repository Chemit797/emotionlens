# Report Plan v3 — EmotionLens 课程报告

> 2026-06-21 | 老师要求七章标题不动，内容叙事按 model-first 来写
> 核心原则：讲"我们训了什么模型、怎么训的、训出来效果怎么样"

---

## 0. 根本定位

### 叙事逻辑
> "我们训了一个 ResNet18 模型，用了 FERPlus 软标签 + RAF-DB 真实人脸 + 自己拍的 1,123 张。训练过程中做了六轮改进，每次只动一个东西——从 11.6% 一路优化到 64.5%。最后把它部署成了一个实时应用。"

模型是主角。训练过程中的六轮改进只是"我们怎么把模型从 11.6% 搞到 64.5% 的"。

### 标题用老师要求的，内容讲模型
- Chapter 5 标题是 `Implementation`，但内容讲的是"我们怎么实现这个模型的"
- Chapter 6 标题是 `Results and analysis`，但内容讲的是"我们的模型表现怎么样"
- 这两件事不矛盾

### 具体要求全部覆盖
- ✅ Calculate loss, fine tune and reduce loss → 5.2(a)(b)
- ✅ Adjust parameters, fine tune parameters → 5.2(c)(g)
- ✅ Implement gradient, slicing, sampling → 5.2(b)(d)
- ✅ Explain collection of dataset → 4.2
- ✅ Provide justification of using specific algorithm → 5.1, 5.2
- ✅ Describe the working of each section of code → 5.2 每小节
- ✅ Clearly identify through comments the URL of where ideas and code were found → 代码注释要求
- ✅ Report structure: Front page / Abstract / Introduction / Literature review / Data set / Implementation / Results and analysis / Conclusion → 严格遵循

---

## 1. 封面 + 目录 (2p)

标题：**EmotionLens: A Cross-Domain Facial Expression Recognition Model**
组号 + 组员 + 日期

---

## 2. Chapter 1: Abstract (0.5p, ~180 words)

**只讲事实，5 句。**

1. 我们训了一个模型：ResNet18 backbone，训练数据来自三个源——FERPlus（10 人投票软标签）、RAF-DB（真实场景人脸）、自采集数据（1,123 张，6 名组员）。
2. 训练过程中，我们做了六轮改进，每轮只动一个变量：架构（VGG16→ResNet18）、标签质量（硬标签→10 人软标签）、数据多样性（单源→多源）。
3. 最终模型：FERPlus 测试集 89.0%，自采集真实人脸 64.5%——相比最初的 FER2013+VGG16 起点（11.6%），提升了 53 个百分点。
4. 标签从硬到软贡献了最大的一步（+35pp），其次是加入真实场景数据恢复了对负情绪的识别（+20pp），换架构几乎没作用（−2pp）。
5. 模型部署为 EmotionLens 实时 Web 应用，5 个分析镜头消费同一个推理引擎——引擎是核心，镜头可以随便换。

---

## 3. Chapter 2: Introduction (0.7p)

### Para 1 — 背景 (2-3 句)
表情识别有什么用——课堂专注度、人机交互、在线监考、心理健康筛查。自动化意味着规模，模型可以 24/7 跑，标准一致。

### Para 2 — 问题 (3-4 句)
问题是：在 lab 数据集上训出来的模型，一碰到真实人脸就崩。标准测试集上动辄 85%+，但换成没见过的真人脸，可能跟瞎猜差不多。这就是 domain gap——模型学的是数据集的 shortcut，不是真正的表情特征。

### Para 3 — 起点 (2-3 句)
我们从课程作业列表上建议的 FER2013 + VGG16 起步。老老实实复现：FER2013 测试集 69.5%，但我们自己拍的 1,123 张真实人脸只有 11.6%。58 个百分点的差距——这就是我们的出发点。

### Para 4 — 我们干了什么 (3-4 句)
我们训了一个基于 ResNet18 的模型，数据用了 FERPlus 的 10 人投票软标签 + RAF-DB 的真实场景人脸 + 6 名组员自采集的 1,123 张。训练过程中做了六轮迭代改进，每次只改一个东西，最终在真实人脸上达到 64.5%。模型部署成了 EmotionLens——一个实时 Web 应用，5 个分析镜头可以随时替换，不改引擎一行代码。

### Para 5 — 报告结构 (1 句)
Section 3 介绍我们选用的数据和架构背景；Section 4 讲数据和预处理；Section 5 讲模型怎么训的、训练过程中做了哪些改进；Section 6 展示模型效果；Section 7 总结。

---

## 4. Chapter 3: Literature Review (1p)

**定位**：老师要求的章节。但不是堆论文——写成"我们做模型之前，看了哪些已有的工作和数据，这些怎么帮助了我们的选择"。

### 3.1 数据：为什么选 FERPlus + RAF-DB (0.3p)
- FER2013：第一个大规模 FER 数据集，35,887 张 48×48 灰度图。但每张只有一个人标，噪声大。
- FERPlus：跟 FER2013 完全一样的图片，但每张由 10 个人独立标注→软标签概率分布。标注噪声大幅降低。
- RAF-DB：~15,000 张真实场景人脸，比 FERPlus 的网上小图更多样。弥补 lab 数据的偏向。

### 3.2 架构：为什么选 ResNet18 (0.3p)
- VGG16：138M 参数，堆叠卷积，经典但重。
- ResNet：残差连接→梯度走短路→训得深不崩。ImageNet 预训练权重可以迁移到 FER。
- EfficientNet-B0：NAS 搜出来的轻量架构，5.3M 参数。但我们实测同条件下只到 41.0%，可能 35 epoch 不够它收敛。
- **我们三个都跑了，ResNet18 在参数效率、训练稳定性、最终准确率之间最均衡。**

[可选插图: ResNet 残差块示意图 — He et al., 2016 Fig. 2]

### 3.3 训练技巧 (0.3p)
- mixup：随机混合两张图→模型见过更多"中间态"→对 domain shift 更鲁棒
- Label smoothing：把硬标签软化→防止过自信→泛化更好
- AdamW：解耦权重衰减的 Adam→比 SGD 收敛快，比普通 Adam 泛化好

---

## 5. Chapter 4: Data Set (1.5p)

### 4.1 公开数据集 (0.5p)
- **FER2013**：35,887 张，48×48 灰度，7 类硬标签，CSV。只用在最初的基线模型。
- **FERPlus**：35,803 张，同上图片但 10 人投票软标签，parquet。8 类→我们丢弃 contempt→统一为 7 类。
- **RAF-DB**：~15,344 张真实场景人脸，7 类硬标签。对齐好、条件多样。

### 4.2 自采集数据 (0.5p)
- 6 名组员，各自笔记本摄像头，教室/宿舍室内自然光，距屏幕 40-60cm
- 每种表情录短视频→脚本抽帧→弱模型预标→**人工逐帧审核**
- 最终 1,123 张，7 类，80/20 分层随机划分
- [图: per_user_emotion_bars.png — 每人每种表情的分布]

### 4.3 预处理 (0.3p)
- 全部统一到 224×224 RGB，ImageNet 归一化
- FER2013/FERPlus：48×48 灰度→resize→3 通道复制
- RAF-DB：RGB→直接 resize
- 自采集：原始帧→MTCNN 人脸检测裁剪→resize

### 4.4 数据统计 (0.2p)

| Source | Train | Val | Test | Total |
|--------|-------|-----|------|-------|
| FER2013 | 28,709 | 3,589 | 3,589 | 35,887 |
| FERPlus | 28,709 | 3,546 | 3,546 | 35,803 |
| RAF-DB  | ~12,000 | — | ~3,000 | ~15,344 |
| Self    | 899 | 224 | — | 1,123 |

---

## 6. Chapter 5: Implementation (4p) ⭐ 核心章节 — 标题老师要求，内容讲模型怎么实现的

**这一章回答**：我们选了什么架构 → 用了什么训练配方 → 训练过程中做了六轮改进 → 最后模型怎么部署的。

**同时覆盖老师所有技术要求**：loss 计算与优化（5.2a/b）、参数调整（5.2c/g）、gradient/slicing/sampling（5.2b/d）、算法选择的 justification（5.1, 5.2 全文）、代码各部分的 working（5.2 每小节）。

---

### 5.1 架构选择 (1p)

我们从三个候选里选。在完全相同的条件下（FERPlus 数据、同一训练配方、35 epochs）跑了对比：

| Architecture | Params | FERPlus Test | Self Data |
|-------------|--------|:---:|:---:|
| VGG16 | 138M | — | 11.6%* |
| EfficientNet-B0 | 5.3M | 41.0% | 19.4% |
| **ResNet18** | **11.2M** | **79.9%** | **44.4%** |

*VGG16 训的是 FER2013；EfficientNet 和 ResNet18 训的是 FERPlus。

VGG16 太重在笔记本上推理慢，EfficientNet-B0 在我们的训练配方下效果不好。ResNet18 的核心优势是残差连接——梯度可以通过短路路径传播，训 35 epoch 稳定收敛。ImageNet 预训练的特征也能迁移到表情任务上。

[插图: ResNet 残差块 — He et al. 2016 Fig. 2]
[插图: VGG vs ResNet 结构对比 — 可选]

最终配置：
```
torchvision.models.resnet18(weights="IMAGENET1K_V1")
→ 替换 FC: Linear(512, 7)
→ 输入: 224×224 RGB, ImageNet normalization
→ 可训练参数: 11,180,103
```

### 5.2 训练配方 (1.5p)

每种技术简述：是什么、为什么用、参数值。

**(a) Loss Function**

```
硬标签数据 (FER2013, RAF-DB) → CrossEntropyLoss
  L = −log p(y_true)
  问题：标注者标错了，loss 还逼着模型 100% 信那个标错的标签

软标签数据 (FERPlus) → Soft CrossEntropy
  L = −Σ_k p_voter(k) · log p_model(k)
  p_voter(k) = 10 个人里有几个投了第 k 类 / 10
  例: 7 人标 happy, 1 人 neutral, 1 人 surprise, 1 人 contempt
      → 目标分布 [0.7, 0.1, 0.1, ..., 0.1]，不是 [1, 0, 0, ..., 0]
  好处：模型不需要 100% 确定 → 对标注噪声更鲁棒
```

**(b) Optimizer + Mixed Precision**

```
AdamW: lr=3e-4, wd=1e-4, β1=0.9, β2=0.999
  → 解耦 weight decay，有限 epoch 内收敛比 SGD 稳

AMP (torch.amp.GradScaler):
  → 前向 FP16 加速 ~40%，反向自动 scale/unscale 梯度
  → RTX 4060 8GB 上 batch_size=128 刚好不爆显存
```

**(c) LR Schedule**

```
CosineAnnealingLR: T_max=35, lr 从 3e-4 平滑衰减到 0
  → 不用手动调衰减节点，最后几个 epoch 探索更精细的 minima
```

**(d) 类别平衡采样**

```
WeightedRandomSampler: w_class = 1/count(class)
  → FER2013/FERPlus 严重不平衡（happiness 7000+, disgust 400+）
  → 少数类过采样，多数类欠采样，每 batch 梯度来自均衡分布
```

**(e) 正则化**

```
Label Smoothing (ε=0.1):
  [0,0,1,0,0,0,0] → [0.015,0.015,0.9,0.015,0.015,0.015,0.015]
  → 防止过自信，测试时更鲁棒

mixup (α=0.2):
  x̃ = λ·xᵢ + (1-λ)·xⱼ, ỹ = λ·yᵢ + (1-λ)·yⱼ
  → 模型见过更多"中间态"，对 domain shift 更鲁棒
```

**(f) 数据增强**

```
RandomHorizontalFlip(0.5), ColorJitter(0.2, 0.2, 0.2), RandomErasing(0.1)
测试时: 水平翻转取平均
```

**(g) 超参数总表**

| Parameter | Value | Why |
|-----------|-------|-----|
| Optimizer | AdamW | 收敛快 + 解耦 weight decay |
| LR | 3e-4 | 比 1e-4 收敛更快且不震荡 |
| Weight decay | 1e-4 | 标准值 |
| LR schedule | CosineAnnealing, T=35 | 平滑，无需手动调 |
| Epochs | 35 | val_acc 在 30-35 附近饱和 |
| Batch size | 128 (VGG16: 64) | VGG16 显存放不下 128 |
| Loss | CE / Soft CE | 跟标签类型匹配 |
| Label smoothing | 0.1 | 标准值 |
| Mixup α | 0.2 | 保守混合 |
| Hardware | RTX 4060 Laptop 8GB | 笔记本训的 |

---

### 5.3 训练过程中的六轮改进 (1.2p) ⭐ Implementation 的核心——讲我们怎么一步步把模型搞好的

**叙述逻辑**：不是"设计了实验去验证假设"。是"训模型的过程中，发现不行就改一个东西再看，一共改了六轮"。每次只改一个变量不是实验方法论，是我们的工程纪律。

---

**第一轮：从课程基线起步**

我们先按课程作业列表上建议的路线，用 FER2013 + VGG16 训了一个基线模型。

```
VGG16 + FER2013 硬标签:
  FER2013 测试集: 69.5%
  自采集真实人脸: 11.6%  ← 根本不能用
```

这个结果不意外——FER2013 的图片是 48×48 灰度网图，标注是一个人标的，跟我们的真实人脸差太远了。但这给了我们一个明确的起点。

**第二轮：试一下换个更好的架构**

第一个想法——是不是 VGG16 太老了？换 ResNet18 试试。其他什么都不改：同样的 FER2013 数据、同样的硬标签、同样的训练参数。

```
ResNet18 + FER2013 硬标签:
  FER2013 测试集: 69.9%  (跟 VGG16 几乎一样，+0.5pp)
  自采集真实人脸:  9.2%  (还更差了，−2.3pp)
```

换架构没解决问题。ResNet18 比 VGG16 先进，但在真实人脸上表现甚至更差。这说明瓶颈不在架构——数据本身有问题。

**第三轮：换标签——从一个人标的硬标签换成 10 个人投票的软标签**

FERPlus 跟 FER2013 是完全一样的图片，区别只是每张图有 10 个人独立标注，标签变成了概率分布。我们把数据从 FER2013 CSV 换成 FERPlus parquet，loss 也相应地从硬 CE 换成软 CE。架构保持 ResNet18 不变。

```
ResNet18 + FERPlus 软标签:
  FERPlus 测试集: 79.9%
  自采集真实人脸: 44.4%  ← 从 9.2% 涨了三倍多！(+35.2pp)
```

这是最大的一步提升。10 个人投票的标签比一个人标的质量高太多了——不是架构不行，是原来的标签太吵。

**第四轮：加数据——把真实场景的人脸加进来**

FERPlus 的标签好了，但图片还是 48×48 灰度网图。模型没见过真实世界的高清人脸，所以在我们的自采集数据上仍然只有 44.4%。我们把 RAF-DB（15,000 张真实场景人脸）和我们自己拍的 1,123 张加进训练集。架构保持 ResNet18。

```
ResNet18 + FERPlus + RAF-DB + Self:
  FERPlus 测试集: 89.0%
  自采集真实人脸: 64.5%  ← 又涨了 20pp (+20.1pp)
```

更关键的是负情绪：sadness 从 0%→53%，disgust 从 0%→53%，fear 从 0%→60%。加了真实场景数据后，模型第一次学会了在真人脸上认这些表情。

**第五轮：跨域评测——看看训出来的模型在不同测试集上表现怎么样**

我们把四轮下来的模型（VGG16+FER2013 / ResNet18+FER2013 / ResNet18+FERPlus / ResNet18+F+R+S）放到三个不同的测试集（FERPlus / FER2013 / Self）上统一跑了一遍。

每个模型在自己训练的领域表现最好——这很正常。重要的是只有多源训练的 F+R+S 在真实人脸上能用。额外的参考：EfficientNet-B0（同条件训 FERPlus）只到 41.0%，一个开源 FER 库（hustvl/fer）在 self data 上 52.6% 但 disgust=0%、anger=1.8%——跟我们早期 VGG16+FER2013 一样的模式。

**六轮改进的累计效果**：

```
11.6% ──→ 9.2% ──→ 44.4% ──→ 64.5%
  基线    换架构    换标签     加数据
           −2.3pp  +35.2pp   +20.0pp
           没用     最关键     第二关键
```

每次只改一个东西，所以每一步的效果是能说清楚的。最终从 11.6% 到 64.5%，净提升 52.9pp。

[图: control_variable_chain.png — 这张图也是"模型改进过程"，不是"实验结果"的标题]

---

### 5.4 模型部署 (0.3p)

模型训好之后，我们把它包成了一个实时 Web 应用 EmotionLens。

```
后端: FastAPI + uvicorn + WebSocket
  摄像头帧 → YuNet 人脸检测 → ResNet18 推理 → EMA 时间平滑 → 输出 7 维概率向量
  5 个分析镜头各自消费同一条推理流，互不干扰

前端: 原生 HTML/CSS/JS + Swiper.js 轮播
  群体统计 / 情绪预警 / 时间线 / 演讲紧张度 / 模仿游戏 / + Build Your Own Lens



5 个镜头只是例子。核心是推理引擎——输出裸 JSON，任何人都能写新镜头，不碰模型代码。

[可选插图: 系统框图 — Webcam → Engine → Lenses]

---

## 7. Chapter 6: Results and Analysis (5p) ⭐ 标题老师要求，内容展示模型表现

**策略**：图多字少。每张图占 ~1/3 页，下面 3-4 行 bullet。说清楚：看什么 → 看到什么 → 说明什么。

### 6.1 模型改进过程
- 图: control_variable_chain.png（四步瀑布图）
- 四步、每步 Δ、总 +52.9pp

### 6.2 架构对比
- 图: architecture_ablation.png（VGG16 vs ResNet18，双测试集柱状图）
- 同数据同配方，架构从 VGG16 换成 ResNet18：lab 测试集 +0.5pp，real faces −2.3pp

### 6.3 逐类表现分析 ⭐ 展开
- 图 1: per_class_recall_all.png（4 模型 × 7 类）
- 图 2: self_confusion_matrix.png（最终模型）

**数据部分**（bullet）：
- FERPlus-only 模型在 sadness/disgust/fear 上 recall = 0%——完全认不出
- 加上 RAFDB+Self 后恢复到 53-60%
- 混淆矩阵：7 类都有预测分布，不再是 collapse 到一两个类

**分析部分**（一段 ~0.5p）：

> 最终模型在 self test 上整体 64.5%，跟 FERPlus test 上的 89.0% 还有明显差距。但拆开看：neutral (78%) 和 surprise (70%) 并不差，差距集中在 disgust (53%)、anger (56%)、fear (60%) 这几类。
>
> 我们推测这背后有两层原因。第一层：neutral 和 happiness 属于人类生物本能，表达方式跨文化高度一致——笑就是笑。但负面情绪的含义更抽象，不同文化的人理解和表达方式差异很大（Jack et al., 2012, PNAS）。第二层：FERPlus 和 RAF-DB 的标注以西方文化标准为主，而我们的 self test 全部来自亚洲组员。当「西方标准下的 disgust」遇上「亚洲人脸的 disgust」，标签一致性本身就存在结构性的上限。换句话说，这几类情绪的低 recall，可能不是模型没学好，而是标注的文化偏差造成的。

### 6.4 跨测试集表现
- 图: cross_domain_radar.png + 完整数字表
- 每个模型在自己训的那个域最好；只有 F+R+S 在真实人脸上能用

### 6.5 Domain Gap
- 图: domain_gap.png（lab acc vs self acc 配对柱状图）
- 从 58pp gap 缩到 25pp：加真实场景数据，lab↔real 差距缩小一半以上

### 6.6 各因素贡献
- 图: contribution_breakdown.png
- 架构 −2.3pp | 标签质量 +35.2pp | 数据多样性 +20.0pp = 净 +52.9pp

### 6.7 第三方 FER 库对比 (0.2p，纯文字)
开源 FER 库 (hustvl/fer) 在我们的 self data 上：52.6% 总准确率，disgust=0%，anger=1.8%。跟 VGG16+FER2013 一样的模式——neutral 还行，负情绪为零。两个独立实现、同源数据、同样问题：不是我们写得烂，是 FER2013 数据本身造不出能泛化的模型。

---

## 8. Chapter 7: Conclusion (0.5p)

**两段收束。**

### Para 1 — 训出了什么模型
我们用 ResNet18 作 backbone，FERPlus 软标签 + RAF-DB + 自采集 1,123 张作训练数据，训了一个跨域表情识别模型。从 FER2013+VGG16 基线（真实人脸 11.6%）开始，经过六轮训练改进，最终模型在 FERPlus 测试集上 89.0%，在我们的自采集真实人脸上 64.5%。52.9pp 的提升中，标签质量贡献了最大的一步（+35.2pp），数据多样性贡献了第二步（+20.0pp），换架构贡献几乎为零（−2.3pp）。

### Para 2 — 这意味着什么
下次有人要做 FER：预算有限就先把标签质量搞好（硬标签→多人投票软标签是回报最高的操作）；有余力再加多样化的真实场景数据。不要指望换个新架构就能解决问题——我们试过了，没用。模型部署成 EmotionLens 之后，5 个镜头只是举例，推理引擎才是可复用的部分——换镜头不改引擎。

---

## 9. References (1p)

12 篇：

1. Ekman & Friesen (1971) — 7 basic emotions
2. Goodfellow et al. (2013) — FER2013
3. Simonyan & Zisserman (2014) — VGG
4. He et al. (2016) — ResNet
5. Barsoum et al. (2016) — FERPlus
6. Zhang et al. (2016) — MTCNN
7. Szegedy et al. (2016) — Label smoothing
8. Zhang et al. (2018) — mixup
9. Li & Deng (2019) — RAF-DB
10. Tan & Le (2019) — EfficientNet
11. Loshchilov & Hutter (2019) — AdamW
12. Jack et al. (2012) — Cross-cultural facial expressions (PNAS)

---

## 10. 图表清单

| # | 文件名 | 内容 | 位置 |
|---|--------|------|------|
| 1 | control_variable_chain.png | 四轮改进瀑布图 | 6.1 |
| 2 | architecture_ablation.png | VGG vs ResNet 对比 | 6.2 |
| 3 | per_class_recall_all.png | 4 模型 × 7 类 recall | 6.3 |
| 4 | self_confusion_matrix.png | 最终模型混淆矩阵 | 6.3 |
| 5 | cross_domain_radar.png | 跨测试集雷达图 | 6.4 |
| 6 | domain_gap.png | lab vs self gap | 6.5 |
| 7 | contribution_breakdown.png | 各因素贡献分解 | 6.6 |
| 8 | per_user_emotion_bars.png | 每人表情分布 | 4.2 |
| 9 | ResNet 残差块示意图 | He et al. 2016 Fig. 2 | 5.1 |

---

## 11. 页数预算

| 章节 | 页数 | 图/表 |
|------|:---:|:---:|
| 封面 + 目录 | 2p | — |
| Abstract | 0.5p | — |
| Introduction | 0.7p | — |
| Literature Review | 1p | 可选 1 图 |
| Data Set | 1.5p | 1 表 + 1 图 |
| **Implementation** | **4p** | 2 图 + 1 大表 |
| **Results and Analysis** | **5p** | 6 图 + 1 表 |
| Conclusion | 0.5p | — |
| References | 1p | — |
| **总计** | **16.2p** | 9 图 + 3 表 |

---

## 12. v3 vs v2 核心改动

| # | 改动 | 说明 |
|---|------|------|
| 1 | **七章标题严格用老师要求的** | Abstract / Introduction / Literature Review / Data Set / Implementation / Results and Analysis / Conclusion |
| 2 | **叙事重心从"实验"移到"模型"** | 全文讲"我们训了什么模型"，实验是训练过程中的改进步骤 |
| 3 | 5.3 从 "Experiment Design" → "训练过程中的六轮改进" | 不讲实验矩阵方法论，讲"训模型时怎么一步一步改的" |
| 4 | 老师所有技术要求显式覆盖 | loss/gradient/sampling/justification/code working 全部标注对应章节 |
| 5 | Abstract 不讲"we designed experiments"，讲"we trained a model" | 开场就是模型 |
| 6 | 全文删 "hypothesize"、"this paper"、"contribution"、"we present" | report 不需要学术仪式感 |
| 7 | 控制变量只作为"每次只改一个东西"的工程纪律出现 | 是纪律，不是主题 |
