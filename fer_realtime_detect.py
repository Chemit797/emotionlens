"""Real-time facial expression recognition via webcam using the `fer` library.
UI and Tracking copied from realtime_detect.py"""
import argparse
import time
import numpy as np
import cv2
from fer.fer import FER

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

FER_TO_EMO = {
    'neutral': 'neutral',
    'happy': 'happiness',
    'surprise': 'surprise',
    'sad': 'sadness',
    'angry': 'anger',
    'disgust': 'disgust',
    'fear': 'fear'
}

# 情绪对应颜色（BGR）
EMO_COLOR = {
    "neutral":    (180, 180, 180),  # 灰色
    "happiness":  (0, 255, 100),    # 绿色
    "surprise":   (255, 200, 0),    # 金色
    "sadness":    (200, 80, 0),     # 蓝紫
    "anger":      (0, 0, 240),      # 红色
    "disgust":    (0, 140, 60),     # 深绿
    "fear":       (120, 0, 180),    # 紫红
}


# ── 跟踪 / 平滑超参 ──────────────────────────────────────────────────────
_EMA_A      = 0.30   # EMA 新帧权重（越小越平滑但响应越慢）
_MIN_HITS   = 3      # 连续检测多少帧后才显示框
_MAX_MISS   = 8      # 丢失检测后保持显示的帧数
_IOU_THRESH = 0.25   # 帧间匹配 IoU 阈值


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0


class FaceTracker:
    """多人脸 EMA 跟踪器：稳定框坐标和情绪概率，过滤瞬间误检。"""

    def __init__(self):
        self._tracks = []

    def reset(self):
        self._tracks = []

    def update(self, raw_rects, raw_probs_list):
        used_det, used_trk = set(), set()

        for ti, t in enumerate(self._tracks):
            best_v, best_di = _IOU_THRESH, -1
            for di, r in enumerate(raw_rects):
                if di in used_det:
                    continue
                v = _iou(t['ri'], r)
                if v > best_v:
                    best_v, best_di = v, di
            if best_di >= 0:
                used_det.add(best_di)
                used_trk.add(ti)
                a, nr = _EMA_A, raw_rects[best_di]
                t['rf'] = [a*nr[i] + (1-a)*t['rf'][i] for i in range(4)]
                t['ri'] = [int(v) for v in t['rf']]
                t['probs'] = a * raw_probs_list[best_di] + (1-a) * t['probs']
                t['hits'] += 1
                t['miss']  = 0
                if t['hits'] >= _MIN_HITS:
                    t['ok'] = True

        # 未匹配的旧轨迹：累加 miss 计数，超过阈值丢弃
        surviving = []
        for ti, t in enumerate(self._tracks):
            if ti not in used_trk:
                t['hits'] = 0
                t['miss'] += 1
            if t['miss'] <= _MAX_MISS:
                surviving.append(t)
        self._tracks = surviving

        # 未匹配的新检测：建立新轨迹
        for di, r in enumerate(raw_rects):
            if di not in used_det:
                self._tracks.append({
                    'rf': [float(v) for v in r],
                    'ri': list(r),
                    'probs': raw_probs_list[di].copy(),
                    'hits': 1, 'miss': 0, 'ok': False,
                })

        return [(t['ri'], t['probs'])
                for t in self._tracks if t['ok'] and t['miss'] <= _MAX_MISS]


def draw_panel(frame, faces_probs, face_rects, fps):
    """在帧上绘制检测结果"""
    h, w = frame.shape[:2]

    for probs, (fx, fy, fw, fh) in zip(faces_probs, face_rects):
        # 取 top-1
        top_idx = int(np.argmax(probs))
        top_emo = EMO[top_idx]
        top_conf = float(probs[top_idx])
        color = EMO_COLOR.get(top_emo, (255, 255, 255))

        # 画人脸框
        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

        # 顶栏：情绪 + 置信度
        label = f"{top_emo}  {top_conf:.0%}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (fx, fy - lh - 10), (fx + lw + 8, fy), color, -1)
        cv2.putText(frame, label, (fx + 4, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # top-3 概率条：优先画在右侧，空间不足则画在左侧
        bar_w = 120
        bar_h = 16
        bar_x = fx + fw + 8
        if bar_x + bar_w > w:        # 右侧超出画面，改为左侧
            bar_x = max(0, fx - bar_w - 8)
        order = np.argsort(probs)[::-1][:3]
        for rank, idx in enumerate(order):
            by = fy + rank * (bar_h + 4)
            conf = float(probs[idx])
            cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (60, 60, 60), -1)
            fill_w = int(bar_w * conf)
            cv2.rectangle(frame, (bar_x, by), (bar_x + fill_w, by + bar_h),
                          EMO_COLOR.get(EMO[idx], (200, 200, 200)), -1)
            text = f"{EMO[idx]:>10s}  {conf:.0%}"
            cv2.putText(frame, text, (bar_x + 3, by + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    # 提示
    cv2.putText(frame, "Q=Quit  S=Screenshot",
                (w - 260, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return frame


def main():
    ap = argparse.ArgumentParser(description="Real-time FER using the 'fer' library")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale factor")
    ap.add_argument("--fast", action="store_true", help="Use Haar Cascade instead of MTCNN in fer")
    ap.add_argument("--gpu", action="store_true", help="Try to run on GPU by disabling TFLite mode")
    a = ap.parse_args()

    print("[model] Loading FER library model...")
    # mtcnn=True is more accurate but slower. If --fast is passed, it uses haar cascade.
    # use_tflite=False forces it to use standard Keras models which can utilize the GPU if tensorflow-gpu is installed.
    detector = FER(mtcnn=not a.fast, use_tflite=not a.gpu)
    print(f"[camera] Started (MTCNN={not a.fast}, TFLite={not a.gpu})")
    print("  Q=Quit  S=Screenshot\n")

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print("[error] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_times = []  # 用于平滑 FPS
    frame_count = 0
    tracker = FaceTracker()

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像
        display = frame.copy()
        faces_probs = []
        face_rects = []

        # FER 库需要 RGB 格式的图片
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调用 fer 库进行检测
        detected_faces = detector.detect_emotions(rgb_img)

        raw_rects, raw_probs_list = [], []
        for face in detected_faces:
            box = face['box']
            emotions = face['emotions']
            
            # 将 fer 的情绪结果映射到我们的 7 类顺序中
            probs = np.zeros(7, dtype=np.float32)
            for i, emo_name in enumerate(EMO):
                # reverse lookup to find fer's key
                for fer_k, our_k in FER_TO_EMO.items():
                    if our_k == emo_name:
                        probs[i] = emotions.get(fer_k, 0.0)
                        break
            
            # 归一化（尽管 fer 库通常会给出加和为1的值，这里加一层保险）
            sum_p = np.sum(probs)
            if sum_p > 0:
                probs /= sum_p

            raw_rects.append(box)
            raw_probs_list.append(probs)

        # 传入 EMA 追踪器进行平滑防抖
        confirmed = tracker.update(raw_rects, raw_probs_list)

        if confirmed:
            face_rects  = [r for r, _ in confirmed]
            faces_probs = [p for _, p in confirmed]

        # 计算并绘制结果
        fps = 1.0 / max(time.time() - t0, 0.001)
        fps_times.append(fps)
        if len(fps_times) > 30:
            fps_times.pop(0)
        smooth_fps = np.mean(fps_times) if fps_times else 0

        display = draw_panel(display, faces_probs, face_rects, smooth_fps)

        # 缩放
        if a.scale != 1.0:
            display = cv2.resize(display, None, fx=a.scale, fy=a.scale)

        cv2.imshow("FER Library Realtime — Emotion Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(out_path, display)
            print(f"[screenshot] {out_path}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[bye] Processed {frame_count} frames.")

if __name__ == "__main__":
    main()
