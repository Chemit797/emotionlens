# EmotionLens — 整体改进目标文档（v4 · 界面全英文 · 交给 Claude Code 实现）

> 一句话定位：**EmotionLens —— 一个模型，多重镜头。** 一套实时表情识别引擎，
> 上面挂多个可插拔的「镜头(Lens)」应用；每个镜头消费同一条情绪流，做各自的分析与可视化。
> 本文档 = 在现有可运行版本上的**精确改进目标**。务必忠于原意、按契约实现，不要再出大问题。

---

## 0. 安全须知（最高优先级）★

- M4 需要调用大模型 API。**API key 绝不硬编码、绝不进提交包/git。**
- 从环境变量读取：`EMO_LLM_API_KEY`（base_url、model 放本地配置 `config.local.json`，该文件加入 `.gitignore`、不提交）。
- **没有 key 时，M4 必须自动退回规则文案**，保证程序在没有 key 的机器上也能完整运行（课程评分机不会有 key）。
- 本文档**有意不包含任何真实 key**。

---

## 0.5 语言规则（全局，务必遵守）★

**平台界面一律英文。** 凡是会显示在 UI 上的字符串——标题、按钮、挡位、评语、台词、建议、状态文案、game over 等——**全部用英文**。
本文档里给出的中文是写给实现者的说明，不是 UI 文案；真正上界面的串以本文档标注的英文为准。
LLM 教练建议也要求**输出英文**。

---

## 1. 防杂乱三原则（贯穿全项目）

1. **统一外壳**：顶栏固定显示项目名 **EmotionLens**（tagline: *One model, many lenses.*）+ 当前镜头的**大标题(英文)**；所有镜头共用同一套视觉 token（深色毛玻璃、情绪色、圆角阴影）。
2. **镜头心智模型**：每个应用都是"同一情绪流的一个镜头"，所以再多也统一，不显拼凑。
3. **引擎/镜头分离 + 注册式扩展**：见 §3。加新应用只是注册一个新镜头，绝不动引擎。

---

## 2. 全局改动（所有镜头通用）

- **子界面大标题**：每个镜头面板顶部一个**显眼大标题**（字号 32–40px、粗体，可带该镜头强调色），让用户一眼知道在用哪个应用。
- **字段契约 = 唯一事实源**：后端 `modes.py` 与前端 `app.js` 的字段名**逐字一致**，抽成共享常量（后端 dict / 前端对象），杜绝再次漂移。
- **会话控制协议统一**（M1/M4 这类计时镜头共用）：
  - 上行：`{type:"control", mode, action:"start", duration:<秒>}` / `{action:"stop"}` / `{action:"reset"}`
  - 状态机：`idle → running(倒计时) → [M4: generating] → done →（reset/再来一次回到 idle）`
- **不许动的东西**：检测引擎、M0、整体架构、§6 视觉系统——**只做本文所列改动，不重构**。

---

## 3. 镜头(Lens)抽象与扩展契约（实现可扩展性）

每个镜头实现两件事，注册进一个镜头表即可出现在轮播里：

- **后端聚合器**：`class Lens: id; title; aggregate(emotion_stream, control_state) -> mode_output(dict)`
- **前端面板**：`renderPanel(mode_output)` —— 读取契约字段、更新 UI。

新增一个应用（例如"课堂专注度""直播弹幕情绪墙"）= 新建一个聚合器 + 一个面板 + 注册，**复用同一引擎与同一情绪流**。文档/README 里写明这个扩展点，即"调用我们的模型、随时加新镜头"。

---

## 4. 各镜头详细规格

> 通用：每镜头维护自己的缓冲；参数入 `config.py`；面板顶部有大标题。

### M0 · 实时检测（Live）
保持现状（正常）。面板大标题 **"Live"**。bbox+标签+侧栏 7 维概率条。

---

### L1 · Cafeteria Mood（面板大标题 "Cafeteria Mood"）—— 改为「计时会话」

**交互**：一个**挡位选择**（30s / 1min / 3min / 10min）+ 一个**开关按钮**。
选挡 → 点开关 → 开始倒计时 → 时间到 → 弹**结算卡**。

**算法**：会话窗口内累计所有帧所有脸 → 各情绪出现占比（distribution）→ 主导情绪 → 满意度分 `S = clip((meanV+1)/2,0,1)*100` → 正/中/负占比 → 一句"反馈如何"评语。

**评语规则(verdict, 英文)**：S≥75 `"Very satisfied 👍"`；55–75 `"Satisfied"`；40–55 `"Mixed"`；<40 `"Unsatisfied"`。

**字段契约 `mode_output`**：
```
state: "idle"|"running"|"done"
duration: int            # 选定挡位(秒)
remaining: float         # 倒计时剩余
n_samples: int
result: {                # state=done 时才有
  distribution: {7类: 0-1},
  main_emotion: str,
  satisfaction: 0-100,
  positive_ratio, neutral_ratio, negative_ratio: 0-1,
  verdict: str
}
```
**可视化**：倒计时大数字（running）；结算卡含满意度大数 + 正中负占比环图 + 7 类分布条 + 主导情绪 + 评语。

---

### L2 · CODE RED（面板大标题 **"CODE RED"**）—— 改名 + 实时饼图 + 更敏感 + 爆裂动效 + 台词

**面板大标题**：**"CODE RED"**（直接、有节目效果，可换皮）。

**实时饼图**：始终显示当前情绪占比的**显眼饼图**。

**更敏感的触发**（参数调低，要的就是好触发的节目效果）：
- `N = p_anger + p_disgust`，`r = 0.6·N + 0.4·A`
- 触发（任一）：① 陡升 `r(now) − r(≈0.4s前最近帧) > 0.25`；② 持续 `len(recent)≥3 且最近3帧 r 都 > 0.5`
- 冷却 2.5s；触发时输出 `trigger`。

**爆裂动效 + 台词**：触发瞬间 → 全屏红闪 + 抖动 + 警报音 + 居中弹出一句**英文台词**（从台词池随机）。台词池(`config`, 可增删)，例如：
  `"Draw — take the shot!"`, `"Suspect lunging — restrain!"`, `"High threat — defend now!"`；
  温和备用句：`"Heads up — agitation spike"`（老师对"击毙"措辞敏感时可整池切到温和版）。

**字段契约**：
```
risk_level: 0-1          # r
alarm: bool
trigger: "spike"|"sustained"|null
distribution: {7类: 0-1} # 实时饼图
banner_text: str         # 触发时的台词；未触发为""
dominant: str
```
**报告需保留一句局限声明**（UI 上可不显示）：FER 做威胁预测信度低、有偏见，本镜头为演示性"情绪陡变报警"，非可部署执法工具。

---

### L3 · Audience Reactions（面板大标题 **"Audience Reactions"**）—— 改名 + 时间轴三角点 + 实时条形 + 实时饼图

**面板大标题**：**"Audience Reactions"**。

**布局（自下而上）**：
1. **底部时间轴**：x = 时间。**每当主导情绪发生切换**，就在切换时刻落一个**三角点**（标注新主导情绪、用其情绪色）。
2. 时间轴**上方：实时条形图**——7 类当前概率的条，实时跳动；**最主要情绪的条高亮**，并在旁边用大字显示当前**主导情绪**。
3. 一个**显眼的实时饼图**：累计情绪占比。

**算法**：逐帧对所有脸取平均 → `probs(7)`、`dominant`、`valence`；维护累计 `distribution`；维护 `markers`：当 `dominant` 相对上一个稳定主导发生改变（带去抖：需连续 ≥k 帧确认，默认 k=4）→ 追加 `{t, emotion}`。

**字段契约**：
```
probs: {7类: 0-1}        # 实时条形
dominant: str            # 大字显示
valence: float
distribution: {7类: 0-1} # 实时饼图
markers: [{t, emotion}]  # 三角点(增量发送，前端 append)
elapsed: float
```
（可选保留"导出"：时间轴图 + 分布 + markers 列表。）

---

### L4 · Speech Coach（面板大标题 **"Speech Coach"**）—— 计时会话 + 结算图表 + 大模型建议

**交互**：挡位（10s / 30s / 1min / 3min）+ 开关按钮，同 L1 的状态机。
running 累计指标 → 结束进入 `generating`（调用大模型）→ `done` 显示图表 + 建议。

**指标**：
- `positivity` = V>0.2 的帧占比
- `anxiety` = （`dominant∈{fear,sadness}` 或 `p_fear+p_sadness>0.5`）的帧占比 ← **弃用旧的 `>max-0.1` 歪招**
- `neutral_pct` = neutral 主导帧占比（**务必返回**）
- `expressiveness` = `1 − neutral_pct`
- `timeline`: `[{t, valence, dominant}]`（画结算图）

**结算图表**：会话情绪随时间折线（valence/主导）+ 四项指标卡。

**大模型建议**：把四项指标 + 简要分布喂给 LLM，生成 3–4 句中文教练建议。**详见 §5。无 key → 退回规则文案。**

**字段契约**：
```
state: "idle"|"running"|"generating"|"done"
duration, remaining
result: {                # done 时
  positivity, anxiety, expressiveness, neutral_pct: 0-1,
  timeline: [{t, valence, dominant}],
  advice: str,
  advice_source: "llm"|"rule"
}
```

---

### L5 · Mimic Game（面板大标题 **"Mimic Game"**）—— 按既定逻辑，修干净 bug。所有 HUD 文案英文（Score / Combo / Time / Round / Game Over / Play Again）。

**逻辑**：每题随机一个目标情绪（显示目标 emoji），限时做出。
- 命中（`p_target > 0.6`）：`score += 10 + combo*2; combo += 1;` 立即换下一题。
- 超时/失败：**`combo = 0`**（修复：原来失败也涨），换下一题。
- **结束条件**（修复无限循环）：固定题数（默认 10 题）或总时长（默认 60s）到 → `game_over=true`，显示结算 + "再来一次"。

**字段契约**：
```
target: str
target_emoji: str
p_target: 0-1
score: int
combo: int
time_left: float
round: int
game_over: bool
best: int
```

---

## 5. M4 的大模型接入规格（`backend/advice.py`）

- 配置来源：`key = os.getenv("EMO_LLM_API_KEY")`；`base_url` / `model` 读 `config.local.json`（不提交）。
- OpenAI 兼容 `chat/completions`。（示例：若用 DeepSeek，`base_url=https://api.deepseek.com`、`model=deepseek-chat`；按你的服务商填。）
- 调用（伪代码）：
```python
import os, json, httpx
def gen_advice(metrics: dict) -> tuple[str, str]:
    key = os.getenv("EMO_LLM_API_KEY")
    if not key:
        return rule_based_advice(metrics), "rule"     # 无 key 退回规则文案
    try:
        cfg = json.load(open("config.local.json"))
        sys = ("You are a speech coach. Given the metrics, reply in ENGLISH with 3-4 "
               "specific, encouraging improvement tips. No greetings, no preamble.")
        usr = f"指标：{json.dumps(metrics, ensure_ascii=False)}"
        r = httpx.post(f"{cfg['base_url']}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": cfg["model"],
                  "messages":[{"role":"system","content":sys},
                              {"role":"user","content":usr}],
                  "temperature":0.7, "max_tokens":200},
            timeout=15)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip(), "llm"
    except Exception:
        return rule_based_advice(metrics), "rule"      # 任何失败都退回，绝不崩
```
- `rule_based_advice`：用**英文**阈值文案（anxiety high → `"Slow down and breathe."`；expressiveness low → `"Add more facial expression and energy."`；positivity high → `"Great presence — keep it up."`）。
- **绝不**打印/日志输出 key。

---

## 6. 视觉细节补充（接 v2 §6 设计规范）

- **镜头大标题**：32–40px 粗体，置于面板顶部，可用镜头强调色。
- **计时器（L1/L4）**：running 时超大倒计时数字 + 环形进度；挡位用分段按钮（pill），选中态高亮。
- **结算卡（L1/L4）**：毛玻璃大卡，关键数字最大、图表居中、文字建议在下。
- **L2 爆裂动效**：`@keyframes` 红色边缘脉冲 + 屏幕轻微 shake + 台词放大弹入（cubic-bezier 回弹）+ 警报音。
- **L3 三角点**：用情绪色的小三角标在时间轴，hover 显示"时间 + 情绪"。
- 动画仍只动 `transform/opacity`；UI 60fps 与推理 12fps 解耦插值（v2 §6.8）。

---

## 7. 验收清单（逐项确认，全过才算完成）

- [ ] 五个面板数字/图表**实时更新**，不再恒为默认值。
- [ ] 每个镜头面板有**显眼大标题**。
- [ ] L1：选挡位→开关→倒计时→结算卡（分布+满意度+主导+评语）。
- [ ] L2：改名；**实时饼图常驻**；表情突变能**马上**触发爆裂动效+随机台词；2.5s 冷却。
- [ ] L3：改名；底部时间轴**主导情绪切换落三角点**；上方实时条形高亮主导；实时饼图。
- [ ] L4：选挡位→开关→倒计时→结算图表 + 建议；**有 key 走 LLM、无 key 退规则**，都不崩。
- [ ] L5：命中加分、**失败 combo 清零**、到题数/时长**游戏结束**并结算。
- [ ] **所有 UI 文案为英文**（标题/按钮/评语/台词/建议/状态/结算）。
- [ ] 后端字段与前端读取**逐字一致**且抽成共享常量。
- [ ] **未改动**引擎、M0、架构、视觉系统之外的任何东西。
- [ ] 提交包**不含任何 API key**；`config.local.json` 在 `.gitignore`。

---

## 8. 给实现者的护栏（请严格遵守）

1. 这是**改进 + 修复**，不是重写。**严禁**重构引擎、M0、整体架构。
2. 一切以 §2 / §4 的**字段契约**为唯一事实源，两端逐字对齐并抽常量。
3. **不准硬编码 API key**；M4 无 key 必须可降级运行。
4. 每完成一个镜头，对照 §7 自查并报告通过情况；有不确定先问，别擅自改设计。
