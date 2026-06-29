# EmotionLens 🎭

[**English**](./README.md) | [**中文简体**](./README_zh-CN.md)

![EmotionLens](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688)

EmotionLens 是一个支持多人、实时面部情绪识别的 Web 桌面应用。底层由我们自定义训练的 **EfficientNet-B0** 模型驱动，结合 **YuNet** 提供强大的面部检测能力，能够实时捕捉和分析人群的面部表情。

## 🌟 核心特性
- **🚀 实时处理**：通过 WebSocket 提供流畅的视频流分析（约 12fps）。
- **👥 多人脸追踪**：基于 OpenCV 的 YuNet 引擎，精准追踪画面中的多张人脸。
- **🧠 7类情绪引擎**：在真实人脸数据上训练的强鲁棒性 EfficientNet-B0 模型，支持 7 种情绪分类（`中性`, `高兴`, `惊讶`, `悲伤`, `愤怒`, `厌恶`, `恐惧`）。
- **🎨 5 大互动场景（Lenses）**：内置 5 种不同业务场景的模式，包括食堂满意度调查、突发情绪预警、观影情绪记录仪、演讲教练以及表情模仿游戏。

---

## 🏗️ 设计哲学："一个引擎，多个镜头 (One Engine, Multiple Lenses)"

我们的核心设计理念是将**计算密集的情绪识别引擎**与**轻量级的应用场景镜头 (Lenses)** 解耦。

```text
[ 核心情绪引擎 ] ──► (实时情绪数据流) ──► [ 镜头 1: 演讲教练 ]
                                   ──► [ 镜头 2: 突发情绪预警 ]
                                   ──► [ 镜头 N: 你的自定义场景 ]
```

1. **底层引擎 (Backend)**：获取摄像头每一帧，只做一次人脸检测和 EfficientNet 模型推理，然后广播一个包含边界框和各情绪概率的结构化 JSON 数据流。
2. **应用镜头 (Frontend/Aggregators)**：消费完全相同的情绪数据流，但做出不同的业务解读。例如，“演讲教练”镜头重点关注自信（高兴/中性占比），而“突发情绪预警”镜头只在检测到恐惧或愤怒的瞬时峰值时触发警报。

这意味着，**你不需要重新训练模型，也不需要改动 AI 推理管线**，就能通过添加一个新“镜头”来快速构建全新的业务应用。

---

## 🚀 如何运行

### 1. 安装依赖
请确保你的环境安装了 Python 3.8+。然后安装必要依赖包：
```bash
pip install -r requirements.txt
```
*(注意：为保证 YuNet 正常工作，OpenCV 版本需 >= 4.7.0)*

### 2. 启动服务
你可以直接通过提供的批处理脚本或 Python 命令启动 FastAPI 后端服务：
```bash
# Windows 环境下双击或运行:
cd emotion-app
start.bat

# 或者手动启动:
python main.py
```

### 3. 打开应用
打开浏览器并访问：
```
http://localhost:8000
```
授予摄像头权限后，系统就会开始进行实时流式处理和情绪分析。你可以通过页面底部的滑动组件（Swipe）自由切换不同的场景镜头。

---

## 🛠️ 如何添加你的专属“镜头” (Lens)

得益于 "一个引擎，多个镜头" 的架构，为本项目扩展新功能极其简单。你可以选择纯前端实现，或者编写后端聚合模块。

### 方式一：纯前端 Lens
后端通过 WebSocket 持续不断地推送情绪数据。你只需要在前端接收这些数据并绘制相应的 UI 即可。

1. **监听 WebSocket 数据**:
   在 `emotion-app/frontend/app.js` 中，`ws.onmessage` 函数接收的数据格式如下：
   ```json
   {
     "faces": [
       {
         "box": [x, y, w, h],
         "dominant": "happiness",
         "conf": 0.95,
         "probs": {"happiness": 0.95, "neutral": 0.02, ...}
       }
     ],
     "fps": 12.5
   }
   ```
2. **创建新的 HTML 容器**:
   在 `index.html` 的轮播组件中增加一个新的 `<div class="swiper-slide">`，用于承载你的 UI。
3. **消费数据并渲染 UI**:
   编写一个 JavaScript 函数读取 `faces` 数组并更新你的界面。例如：如果你想做一个“课堂专注度追踪器”，你只需要计算 `faces` 数组中 `neutral` 和 `happiness` 的占比，并将其映射到一个环形进度条上。

### 方式二：后端驱动 Lens
如果你的场景涉及复杂的状态管理（例如将历史数据保存到数据库，或触发邮件报警等），你可以通过在 `emotion-app/backend/lenses/` 添加一个后端的 Python 模块来实现。
1. 新建一个 `my_custom_lens.py` 文件。
2. 编写一个聚合器类来处理原始的推理输出。
3. 将它注册到 `emotion-app/backend/modes.py` 的 WebSocket 分发器中，将处理好的结果推送给前端。

尽情发挥创意，在 EmotionLens 上构建你的专属应用吧！
