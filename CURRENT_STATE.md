# FER-Train 项目概况

**最后更新：2026-06-17**

---

## 当前模型

**ResNet18 7-class**（来自同学 @zhy211423）

| 项目 | 值 |
|---|---|
| 架构 | ResNet18 (torchvision) |
| 训练数据 | FERPlus + RAFDB + Self |
| 类别 | neutral, happiness, surprise, sadness, anger, disgust, fear (7 类，无 contempt) |
| 输入 | RGB 3-channel, ImageNet 归一化 |
| 图像尺寸 | 224×224 |
| 预训练 | 无 (weights=None) |

### 模型文件

| 路径 | 说明 |
|---|---|
| `runs/classmate_model/best.pt` | ★ 当前使用的基础模型 |
| `../emotionz/emotionz/final_outputs/personalized.pt` | 针对 user5 微调版 (user5: 45.2%) |
| `../emotionz/emotionz/final_outputs/best.pt` | 原始副本 |

---

## 核心性能 (Self Data)

**Overall Accuracy: 72.3%** (旧模型 EfficientNet-B0: 44.8%)

| 情绪 | 旧模型 | 新模型 |
|---|---|---|
| neutral | 90.6% | 81.5% |
| happiness | 55.8% | 56.6% |
| surprise | 60.8% | 82.3% |
| sadness | 0% | 53.1% |
| anger | 0.9% | 65.2% |
| disgust | 0% | 72.0% |
| fear | 4.3% | 70.7% |
| contempt | 0% | — (已移除) |

### 按用户

| User | 旧模型 | 新模型 | 图片数 |
|---|---|---|---|
| user1 | 58.4% | 57.3% | 447 |
| user2 | 28.7% | 97.2% | 106 |
| user3 | 75.0% | 95.8% | 120 |
| user4 | 26.3% | 100% | 116 |
| user5 | 17.5% | 15.7% | 115 |
| user6 | 33.3% | 93.2% | 219 |

---

## 关键文件说明

### 模型推理 & 前端

| 文件 | 用途 | 如何运行 |
|---|---|---|
| `realtime_detect.py` | 摄像头实时表情识别 | `python realtime_detect.py` (Q 退出, S 截图) |
| `realtime_detect.py --no-face-detect` | 整帧模式(不检测人脸) | `python realtime_detect.py --no-face-detect` |

### 评估脚本

| 文件 | 用途 | 输出位置 |
|---|---|---|
| `per_user_emotion.py` | 用户×表情 细分准确率(表格+热力图+柱状图) | `runs/self_eval/per_user_emotion_*` |
| `eval_self_data.py` | **全套评估**: 混淆矩阵、原型、domain gap、雷达图、FERPlus 对比 | `runs/self_eval/` (15 个文件) |
| `eval_classmate_model.py` | 新旧模型对比 (一次性使用，可删) | `runs/self_eval/classmate_*` |
| `make_report_figures.py` | 报告级图表: 训练曲线、FERPlus 分布、混淆矩阵 | `runs/training_curves.png` 等 |

### 数据目录

| 目录 | 内容 |
|---|---|
| `self/user1~user6/` | 组员自拍数据 + prelabels.csv |
| `user2_data/` | user2 独立目录 (raw/user2/...) |
| `data/ferplus.parquet` | FERPlus 数据集 |
| `runs/self_eval/` | **所有图表输出目录** |

### 旧模型 (已弃用)

| 路径 | 说明 |
|---|---|
| `runs/efficientnet_b0_20260615_235918/best.pt` | 旧 EfficientNet-B0 8-class |
| `runs/efficientnet_b0_20260615_235918/metrics.json` | 旧模型训练历史 |

---

## 技术细节 (写新脚本时注意)

### 模型加载

```python
import torch, torch.nn as nn
import torchvision.models as tvm

ckpt = torch.load("runs/classmate_model/best.pt", map_location=device)
m = tvm.resnet18(weights=None)
m.fc = nn.Linear(m.fc.in_features, 7)          # 7 类
m.load_state_dict(ckpt["model_state"])          # 注意 key 是 "model_state"
m = m.to(device).eval()
```

### 预处理 (RGB + ImageNet 归一化)

```python
from torchvision import transforms
from PIL import Image

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 从 cv2 BGR 图像
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_pil = Image.fromarray(img_rgb)
tensor = preprocess(img_pil)  # shape: (3, 224, 224)
```

### 类别标签

```python
EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
# 7 类，无 contempt。旧数据中的 contempt 标签会被自动过滤(NaN)。
```

### 同学原始目录

```
../emotionz/emotionz/
├── final_outputs/
│   ├── best.pt              ← base model (我们用的)
│   ├── personalized.pt      ← user5 微调版
│   ├── history.json         ← 训练历史 (30 epochs)
│   └── personalized_report.json
├── server.py                ← FastAPI + WebSocket 推理服务
├── app.js                   ← 前端 (漂亮的 Web UI)
├── index.html
└── style.css
```

---

## 已知问题 & TODO

1. **user5 偏低 (15.7%)** — 可用 `personalized.pt` 提到 45.2%，但会牺牲 neutral 准确率
2. **contempt 不支持** — 同学模型是 7 类的，含 contempt 的旧数据被丢弃了
3. **同学 Web 前端未迁移** — `../emotionz/` 下的 `server.py` + `app.js` 是漂亮 Web UI，可考虑整合
4. **realtime_detect.py 人脸检测较慢** — 用 Haar Cascade，可换成 MTCNN (如同学的做法)
