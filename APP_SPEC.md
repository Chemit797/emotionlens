# 实时表情识别 · 多场景应用 — 目标与实现规格说明书

> 版本：v2（7 类 + 视觉设计规范）
> 定位：在已训练好的 **7 类**表情识别模型之上，构建一个**含前端网页**的实时应用。
> 设计哲学：**一个引擎，多个镜头** —— 底层一套实时情绪识别管线，上层多个"应用镜头"各自消费同一条情绪流。

---

## 0. 一句话目标

打开网页 → 摄像头实时画框识别 7 类表情（多人）→ 通过左右滑动的轮播切换不同"应用镜头"（食堂满意度 / 突发情绪预警 / 观影情绪记录仪 / 演讲教练 / 表情模仿游戏），每个镜头把同一条情绪流加工成各自的指标与可视化。

模型已就绪：EfficientNet-B0，灰度单通道，**直训为 7 类输出**（已去除 contempt）：
`[neutral, happiness, surprise, sadness, anger, disgust, fear]`，`num_classes=7`。
> 引擎直接消费 7 维 softmax，无需任何丢维/重归一化处理。

---

## 1. 系统架构

```
┌──────────────────────── 浏览器（前端）────────────────────────┐
│  getUserMedia 取摄像头帧                                      │
│        │  每帧降到 ~480p、JPEG、≈12fps                         │
│        ▼                WebSocket                            │
│   发送帧  ──────────────────────────────►  接收结果 ──┐       │
│                                                       ▼       │
│   <video> + <canvas overlay>：画框/标签/emoji   各镜头面板    │
│   底部轮播(swipe)切换镜头：仪表盘/曲线/饼图/告警/游戏          │
│   UI 以 60fps(rAF) 渲染，结果以 12fps 到达 → 插值过渡         │
└───────────────────────────────────────────────────────────────┘
                          ▲   │
              结果 JSON    │   │  帧(JPEG)
                          │   ▼
┌──────────────────────── 服务端（Python / FastAPI）────────────┐
│  ENGINE（核心引擎，每帧一次）                                  │
│   YuNet 人脸检测 → (可选)关键点对齐 → 灰度+resize+归一化        │
│   → EfficientNet-B0(7类) 推理 → softmax → 时间平滑(EMA)        │
│   → 产出「情绪流」: 每脸 {bbox, probs[7], dominant, conf}       │
│                          │                                    │
│  MODES（镜头/聚合器，消费情绪流）                              │
│   M1 食堂满意度 · M2 突发情绪预警 · M3 观影记录仪              │
│   M4 演讲教练 · M5 表情模仿游戏                                │
└───────────────────────────────────────────────────────────────┘
```

**为什么这样分工**：检测+识别只做一次（引擎），所有镜头共用结果；切镜头只是切换聚合器，不重算推理。每个镜头是一段独立、可单独 justify 的算法 → 正好对应评分表的 "level of effort" 与 "justification"。

---

## 2. 技术栈与选型理由

| 部件 | 选型 | 理由 |
|------|------|------|
| 后端 | FastAPI + uvicorn + WebSocket | 异步、低延迟、单文件可起；本地 demo 无网络延迟 |
| 模型推理 | 直接加载 `best.pt`（PyTorch，7 类） | 复用已训练模型，**零转换风险**；聚合算法也用 Python，一种语言到底 |
| 人脸检测 | **YuNet**（`cv2.FaceDetectorYN`，opencv≥4.7） | 远准于 Haar、支持多人、附 5 关键点；体积小（onnx ~230KB），无需额外重依赖 |
| 人脸对齐(可选) | 用 YuNet 双眼关键点做仿射对齐 | 拉正人脸 → 识别更稳，是一个加分项 |
| 前端 | 原生 HTML/CSS/JS（无构建步骤） | `main.py` 一键起服务即可，免 Node 工具链；课程提交友好。**颜值靠 §6 设计规范保证，与技术栈无关** |
| 轮播 | **Swiper.js**(CDN) | 原生触感、惯性、回弹 → "滑动自然"的关键 |
| 图表 | Chart.js(CDN) | 时间曲线 + 饼图，自带平滑过渡 |
| 通信 | WebSocket（帧上行 / 结果下行） | 实时双向，比轮询 HTTP 流畅 |

> 可选进阶（写进报告"未来工作"即可，本期不强求）：把模型导出 ONNX 用 onnxruntime-web 在**浏览器内**推理 → 视频不出本机，主打隐私，是对比云 API 的卖点，也是唯一"买域名后能真正上线"的路子（可白嫖 GitHub Pages / Vercel 静态托管）。

---

## 3. 核心引擎规格（`backend/engine.py`）

### 3.1 逐帧管线

1. 解码上行 JPEG → BGR 图。
2. **YuNet 检测**：得到若干人脸 `(x,y,w,h)` + 置信度 + 5 关键点；过滤 `conf < DET_CONF`（默认 0.6）与过小人脸（`min(w,h) < MIN_FACE_PX`，默认 60）。
3. **(可选)对齐**：用双眼关键点把脸旋正、裁到固定框。
4. **预处理**：转灰度 → resize 到训练输入尺寸（与训练一致，128 或 224）→ 归一化 `(x/255-0.5)/0.5` → 单通道张量。
5. **推理**：EfficientNet-B0(7 类) → logits → `softmax` → `probs[7]`。
6. **时间平滑（EMA）**，按人脸 track id 维护：
   `p_smooth = γ · p_prev + (1-γ) · p_now`，γ 默认 0.6。消除逐帧抖动，标签更稳。
7. **输出情绪流**：每脸 `{track_id, bbox, probs(7), dominant=argmax, conf}`。

### 3.2 多脸跟踪（轻量）

用 IOU 贪心匹配上一帧 bbox 给每脸一个 `track_id`（维持 EMA 与各镜头的时序）。丢失超过 N 帧则销毁。够用即可，不引入重型 tracker。

### 3.3 共享情感映射表（`config.py`，所有镜头复用，**可调**）

每类情绪映射到 **valence（正负效价）** 与 **arousal（唤醒度）**：

| 情绪 | valence | arousal |
|------|--------:|--------:|
| happiness | +1.0 | 0.5 |
| surprise | +0.2 | 0.8 |
| neutral | 0.0 | 0.1 |
| sadness | −0.6 | 0.3 |
| fear | −0.7 | 0.85 |
| disgust | −0.7 | 0.6 |
| anger | −0.9 | 0.9 |

> 这是一组**可调的设计选择**（依据情感心理学的二维 valence-arousal 模型）。报告里需说明取值依据并标注"可调"，避免被当成硬编码魔数。

由 `probs` 得连续标量（每帧每脸）：
- 效价 `V = Σ_e valence_e · p_e` ∈ [−1, 1]
- 唤醒 `A = Σ_e arousal_e · p_e` ∈ [0, 1]

---

## 4. 应用镜头规格

> 通用约定：每个镜头维护自己的滚动缓冲区（按时间或帧）。参数全部进 `config.py`，便于调试与报告里的"调参对比"。

### M0 · 实时基础（默认镜头，始终可见）

- **目标**：展示引擎本身。每张脸画框 + 中文标签 + 对应 emoji + 7 类概率条。
- **可视化**：canvas 叠加框与标签；脸旁浮动 emoji（随 dominant 切换、轻微动画）；侧栏 7 类概率横条实时刷新。
- **emoji 映射**：neutral😐 happiness😄 surprise😲 sadness😢 anger😠 disgust🤢 fear😨。

---

### M1 · 食堂满意度检测

- **目标**：单位时间内估计食客整体满意度与主流观感。
- **输入窗口**：滑动窗 `W_sec`（默认 30s）内所有帧、所有脸。
- **算法**：
  - 满意度分（0–100）：`S = clip( (mean_{faces,frames} V + 1) / 2 , 0, 1) × 100`（把效价 [−1,1] 线性映射到 [0,100]）。
  - 主流观感：窗口内 dominant 的众数；同时给"正/中/负"三分占比 —— 正 = happiness+surprise，负 = anger/disgust/sadness/fear，中 = neutral。
- **可视化**：大号满意度仪表盘 + 正/中/负占比环图 + 满意度随时间折线。
- **参数**：`W_sec`、`DET_CONF`、最小有效脸数 `MIN_FACES`（<它则显示"样本不足"）。
- **边界与失败处理**：进食时天然偏 neutral → 视为中性基线、以 happiness 抬升为满意信号；咀嚼/说话误检 → EMA + 置信度门控；人太少 → 显式"样本不足"，不硬报分。

---

### M2 · 突发情绪预警（演示皮肤：警察缉捕）

- **真实目标（技术本质）**：检测**突发的高唤醒负面情绪**（从平静骤变愤怒/恐惧）并即时报警。"警察对峙、嫌犯突袭"只是演示外壳。
- **输入**：单脸（或主目标脸）的实时 `V/A` 与负面概率。
- **算法**（重点在"骤变"，不在"水平"）：
  - 负面强度 `N = p_anger + p_disgust`
  - 风险分 `R(t) = w1·N + w2·A`（默认 w1=0.6, w2=0.4）
  - **触发条件（任一）**：
    1. 持续高位：`R > R_hi`（默认 0.7）连续 `k` 帧（默认 5）；
    2. **陡升**：`R(t) − R(t−Δ) > R_spike`（默认 0.35，Δ≈0.4s）→ 捕捉"突然袭击"前兆。
  - **冷却** `cooldown_sec`（默认 3s）内不重复报警。
- **可视化**：实时 `R(t)` 折线 + 阈值线；触发时红屏脉冲 + 蜂鸣 + "⚠ 高风险：注意自卫"。
- **边界**：打哈欠/大笑误触 → 置信度门控 + 需"陡升或持续"而非单帧；夜间/侧脸检测弱 → 提高 `DET_CONF`。
- **⚠ 报告必写的局限与伦理声明**：基于面部表情"预测攻击/犯罪倾向"在学术上**信度低、且存在偏见与伦理风险**，本模式仅作"突发情绪陡变报警器"的**演示**，不构成可部署的执法工具。诚实写出这点反而踩中评分表对"分析与局限"的要求。

---

### M3 · 观影 / 观众情绪记录仪（建议作为"正经主打"）

- **目标**：实时记录观众情绪随时间的变化，标注剧烈波动点，会后导出报告。
- **输入**：整段会话；逐帧聚合（多人则取平均 + 也可分人）。
- **算法**：
  - 时间轴：每 `t` 记录 `{ts, V, 7类均值概率, dominant}`。
  - **累计分布**：7 类出现占比（饼图，实时更新）。
  - **波动点检测**（标记"名场面"），满足任一记一事件 `{ts, emotion, intensity}`：
    1. 强情绪峰：某非 neutral 类 `p_e > 0.7`；
    2. 急转：`|V(t) − V(t−Δ)| > V_jump`（默认 0.4）。
- **可视化**：实时**效价/各情绪折线** + 实时**情绪占比饼图** + 时间轴上的**波动点标记**。
- **会后导出**：时间曲线 PNG、情绪分布、关键时刻列表（含时间戳）→ 直接可入报告 / 可做"内容测试"卖点。
- **参数**：采样间隔、`V_jump`、强峰阈值、平滑窗。
- **边界**：观影环境暗 → 检测门控；多人时给"群体均值 + 可切单人"。

---

### M4 · 演讲 / 面试教练（新增）

- **目标**：对着摄像头练讲，给出表现反馈。
- **输入**：单人整段会话。
- **指标**：
  - 正向时长占比 `pos_ratio`（V>0.2 的时间比例）；
  - 焦虑占比 `anx_ratio`（fear+sadness 主导的时间比例）；
  - 表现力 `expressiveness`：情绪分布的时间方差/熵（越呆板越低）；
  - 中性占比 `neutral_ratio`。
- **反馈规则（阈值触发文案）**：如 `anx_ratio>0.3 →"紧张偏多，放慢呼吸"`、`expressiveness 低 →"表情偏平，适当增加变化"`、`pos_ratio 高 →"亲和力 good"`。
- **可视化**：自身情绪时间曲线 + 四项指标卡 + 文字建议。

---

### M5 · 表情模仿小游戏（新增，承接 emoji 互动）

- **目标**：把"网页上 emoji 和你互动"做成正经玩法，同时证明 7 类都识别得动。
- **玩法**：屏幕随机给一个**目标 emoji**（某类情绪），`T` 秒内做出对应表情。
  - 得分 = 保持窗口内目标类概率均值 `mean p_{e*}`；`mean p_{e*} > 0.6` 判成功，进入下一题；连击计分。
- **可视化**：大号目标 emoji + 实时进度环（当前 `p_{e*}`）+ 倒计时 + 得分/连击。
- **价值**：demo 极讨喜；天然覆盖全部 7 类，侧面展示模型鲁棒性。

---

## 5. 前端 UI / 交互规格

- **主视图**：全屏 `<video>` + 叠加 `<canvas>`（框/标签/emoji）。
- **镜头轮播**：底部卡片轮播，支持**左右滑动 / 方向键 / 屏上箭头**切换；切换即向后端发 `set_mode`。
- **每镜头面板**：M1 仪表盘+环图、M2 风险曲线+告警、M3 双折线+饼图+时间轴、M4 指标卡+建议、M5 游戏 HUD。
- **会话控制**：M3/M4 有"开始/停止/导出"按钮（控制聚合器缓冲与导出）。
- **状态提示**：摄像头权限、样本不足、连接断开等显式提示。

---

## 6. 视觉设计规范（让它"不丑 + 滑动自然"）★ 本版新增

> 关键认知：简陋感来自**没写样式 + 用浏览器默认控件**，与"原生 HTML/JS"这个技术选择**无关**。把本节写进规格，编码产出的就是带设计的版本，不是裸控件。

### 6.1 基调
深色 + 毛玻璃（glassmorphism）+ 单一强调色。全屏摄像头画面作背景，信息层以半透明毛玻璃卡片浮于其上。

### 6.2 颜色 token（CSS variables，放 `:root`）
```css
--bg:            #0B0E14;                       /* 深底 */
--surface:       rgba(255,255,255,0.06);        /* 玻璃卡片 */
--surface-2:     rgba(255,255,255,0.10);        /* 悬浮态 */
--border:        rgba(255,255,255,0.12);
--text:          #E8ECF1;
--text-dim:      #9AA4B2;
--accent:        #4F8CFF;                        /* 主强调 */
--ok:#34D399;  --warn:#FBBF24;  --danger:#F43F5E;
/* 情绪专属色（框/图表/概率条统一用这套） */
--emo-happiness:#FFC53D; --emo-neutral:#9AA4B2; --emo-surprise:#38BDF8;
--emo-sadness:#6B8AFE;   --emo-fear:#B98BFF;     --emo-disgust:#7CCF6A;
--emo-anger:#FF5C5C;
```

### 6.3 字体
英文 `Inter`/`system-ui`，中文 `"Noto Sans SC"`/`PingFang SC`。字号阶梯：数据大数字 32–40px、标题 18–20px、正文 14px、辅助 12px。数字用等宽 `font-variant-numeric: tabular-nums`（跳动不晃位）。

### 6.4 间距 / 圆角 / 阴影
8px 栅格（4/8/12/16/24）；卡片圆角 16px、按钮/标签 pill 999px；阴影柔和 `0 8px 30px rgba(0,0,0,.35)`。

### 6.5 玻璃卡片组件（统一复用）
```css
.glass{
  background:var(--surface); border:1px solid var(--border);
  border-radius:16px; backdrop-filter:blur(14px) saturate(120%);
  box-shadow:0 8px 30px rgba(0,0,0,.35); padding:16px;
}
```

### 6.6 轮播（Swiper.js）—— "滑得自然"的核心
```js
new Swiper('.modes', {
  effect: 'coverflow',                 // 卡片带纵深，切换有层次
  coverflowEffect:{ rotate:0, depth:120, modifier:1.6, slideShadows:false },
  slidesPerView: 1.15,                 // 露出下一张一角，暗示可滑
  centeredSlides: true, grabCursor:true,
  speed: 420,                          // 切换时长，跟手又不拖
  freeMode:{ enabled:true, momentum:true, momentumRatio:0.6 },
  keyboard:true, mousewheel:{ forceToAxis:true },
  pagination:{ el:'.swiper-pagination', clickable:true },
});
```
触屏惯性、回弹、键盘/滚轮都由 Swiper 接管 —— 这是"滑动自然"最省力的来源。

### 6.7 动画细节（逐项）
- **人脸框不跳**：用 `requestAnimationFrame` 渲染，bbox 对上一帧做线性插值 `b = lerp(b_prev, b_new, 0.4)`；圆角描边 + 顶部标签 pill（颜色取情绪色）。
- **emoji 切换**：`transform: scale()` 带回弹 `cubic-bezier(.34,1.56,.64,1)`；idle 时轻微上下浮动。
- **数字（满意度/得分）**：count-up 缓动到目标值，禁止瞬跳。
- **图表**：Chart.js `animation.duration: 350`；折线 `tension: 0.35` 平滑；饼图占比变化补间。
- **面板切换**：`opacity` + `translateY(12px)` 淡入，200ms `ease-out`。
- **M2 告警**：边缘红色脉冲 `@keyframes`（呼吸式），配合蜂鸣。
- **全局守则**：只动 `transform`/`opacity`（GPU 合成，不触发重排）；过渡统一 200–300ms `ease-out`。

### 6.8 性能与观感解耦（"看起来顺"的根本）
**UI 渲染 60fps（rAF）与推理 12fps 解耦**：识别结果到达只更新"目标值"，渲染层每帧把当前值向目标插值。于是即使推理只有 12fps，框、数字、曲线在视觉上依然丝滑流动，不被推理帧率拖累。这一条比任何美化都更决定"高级感"。

---

## 7. 通信协议（WebSocket）

**上行（浏览器 → 服务端）**
```json
{ "type": "frame", "ts": 1718600000.12, "data": "<base64 jpeg>" }
{ "type": "set_mode", "mode": "recorder" }
{ "type": "control", "action": "start_session" | "stop" | "export" }
```

**下行（服务端 → 浏览器）**
```json
{
  "type": "result", "ts": 1718600000.13,
  "faces": [
    { "track_id": 1, "bbox": [x,y,w,h], "conf": 0.93,
      "dominant": "happiness",
      "probs": { "neutral":0.02,"happiness":0.81,"surprise":0.05,
                 "sadness":0.03,"anger":0.04,"disgust":0.02,"fear":0.03 } }
  ],
  "mode": "cafeteria",
  "mode_output": { "satisfaction": 76.4, "pos":0.6,"neu":0.3,"neg":0.1 }
}
```
`mode_output` 字段随镜头不同（满意度/风险/曲线点/指标/游戏分）。

**性能护栏**：前端**节流到 ~12fps**、每帧降到 ~480p；后端若处理积压则丢旧帧只处理最新，避免延迟堆积。

---

## 8. 项目结构与运行

```
emotion-app/
├── main.py               # 入口：启动 uvicorn + 自动开浏览器
├── requirements.txt
├── backend/
│   ├── server.py         # FastAPI + WebSocket 路由
│   ├── engine.py         # 检测+对齐+推理(7类)+平滑 → 情绪流
│   ├── modes.py          # M1..M5 聚合器
│   └── config.py         # valence/arousal 表、各阈值参数
├── models/
│   ├── best.pt           # 你的 7 类模型（约 20MB，若超 20MB 放 OneDrive）
│   └── face_detection_yunet.onnx
├── frontend/
│   ├── index.html
│   ├── app.js
│   ├── styles.css        # §6 的设计 token 与组件都落这里
│   └── lib/              # Swiper / Chart.js（CDN 或本地）
└── README.txt            # 运行说明（课程要求）
```

- `main.py`：`python main.py` 即起服务并打开 `http://localhost:8000`。
- **提交约束**：Moodle 单文件 ≤20MB；`best.pt` 若超限 → 放 OneDrive/XMU drive，链接写进 cover；`README.txt` 写清依赖与启动步骤。
- **为什么不用买域名/网站**：localhost 已是完整网站，"买"买的只是**地址与可访问性**，不改长相不改功能；且非 localhost 开摄像头**强制 HTTPS**，跑 PyTorch 还要 GPU 服务器持续烧钱。课程只要 `main.py` 本地一键起，买域名零加分。

---

## 9. 里程碑与优先级（对齐 6/25 截止）

**必做（P0，保证可交 + 可 demo）**
1. 引擎跑通（YuNet + best.pt(7类) + 平滑 + 情绪流）。
2. 前端骨架 + §6 设计规范：视频+画框+Swiper 轮播+WebSocket 闭环 + 解耦渲染。
3. M0 实时基础 + **M3 观影记录仪**（正经主打）+ **M1 食堂满意度**。

**加分（P1，有时间再上，可砍）**
4. M5 表情模仿游戏（demo 讨喜，性价比最高，建议优先）。
5. M2 突发情绪预警（+伦理局限段落）。
6. M4 演讲教练；人脸对齐；ONNX 浏览器内推理（未来工作 / 上线路径）。

> 时间紧就守住 P0 三镜头 + M5，足够"功能完整 + 有创意 + 能 demo + 不丑"。

---

## 10. 风险与应对

| 风险 | 应对 |
|------|------|
| WS 帧积压导致延迟 | 前端节流 12fps + 后端只处理最新帧 |
| 多人时算力吃紧 | 限制同时处理人脸数（如 ≤5）；B0 很轻，一般够 |
| 摄像头彩色 vs 模型灰度 | 引擎统一转灰度（与训练一致），无域差 |
| 暗光/侧脸检测差 | 调 `DET_CONF`、`MIN_FACE_PX`；提示补光 |
| 标签抖动 | EMA 平滑 + 置信度门控 |
| 画面卡顿/廉价感 | §6.8 渲染-推理解耦 + 只动 transform/opacity |
| best.pt 超 20MB | 放 OneDrive，cover 附链接 |
| 浏览器摄像头权限 | 首屏引导授权 + 失败提示 |

---

## 11. 与评分表的对应

- **Program Functionality**：实时多人识别 + 多镜头切换，功能闭环。
- **Level of Effort / 雄心**：自训 PyTorch 模型 + 多场景聚合算法 + 带设计的前后端网页 → 明显高于"调库套壳"。
- **Report / justification**：架构选型、YuNet vs Haar、各镜头算法公式与参数、调参对比、混淆矩阵、与现成 FER 库的 baseline 对比、M2 的局限与伦理 → 踩满 "justification + 结果分析"。
- **与 API 的区分点**：离线/隐私、可解释、应用层创新 —— 云 API 黑箱给不了，写进报告即差异化。

---

*下一步：按 P0 顺序编码。可据本规格逐文件实现，或转成给编码 agent 的执行 runbook。*
