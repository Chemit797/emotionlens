"""Real-time facial expression recognition via webcam.
Model: ResNet18 (7-class), trained on FERPlus + RAFDB + self data."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
import cv2

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

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
        """
        raw_rects      : list of (x,y,w,h)
        raw_probs_list : list of np.array，与 raw_rects 等长
        返回已确认的 [(rect_int, smoothed_probs), ...]
        """
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


def load_model(ckpt_path, device):
    """Load ResNet18 7-class model from checkpoint."""
    ckpt_data = torch.load(ckpt_path, map_location=device)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(ckpt_data["model_state"])
    return m.to(device).eval()


def preprocess_face(face_bgr, img_size):
    """输入 BGR 人脸区域，输出 tensor (1, 3, H, W)  RGB ImageNet 归一化"""
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(face_pil).unsqueeze(0)


@torch.no_grad()
def predict_batch(model, faces_tensor, device):
    """批量推理，返回 (N, 7) probs numpy array"""
    if faces_tensor is None:
        return np.array([])
    x = faces_tensor.to(device)
    logits = model(x)
    return torch.softmax(logits, 1).cpu().numpy()


def draw_overlay_fullframe(frame, probs):
    """整帧模式 — 画面中央大字显示"""
    h, w = frame.shape[:2]
    top_idx = int(np.argmax(probs))
    top_emo = EMO[top_idx]
    top_conf = float(probs[top_idx])
    color = EMO_COLOR.get(top_emo, (255, 255, 255))

    # 半透明遮罩顶端
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    # 主预测大字
    cv2.putText(frame, f"{top_emo}  {top_conf:.0%}",
                (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

    # 右侧 top-3
    bar_x = w - 260
    bar_y = 10
    bar_w = 160
    bar_h = 18
    order = np.argsort(probs)[::-1][:3]
    for rank, idx in enumerate(order):
        by = bar_y + rank * (bar_h + 4)
        conf = float(probs[idx])
        cv2.rectangle(frame, (bar_x, by), (bar_x + bar_w, by + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * conf)
        cv2.rectangle(frame, (bar_x, by), (bar_x + fill_w, by + bar_h),
                      EMO_COLOR.get(EMO[idx], (200, 200, 200)), -1)
        text = f"{EMO[idx]:>10s} {conf:.0%}"
        cv2.putText(frame, text, (bar_x + 4, by + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return frame


def draw_panel(frame, faces_probs, face_rects, fps, is_fullframe=False):
    """在帧上绘制检测结果"""
    h, w = frame.shape[:2]

    if is_fullframe and faces_probs:
        draw_overlay_fullframe(frame, faces_probs[0])
    else:
        for probs, (fx, fy, fw, fh) in zip(faces_probs, face_rects):
            # 取 top-1
            top_idx = int(np.argmax(probs))
            top_emo = EMO[top_idx]
            top_conf = float(probs[top_idx])
            color = EMO_COLOR.get(top_emo, (255, 255, 255))

            # 画人脸框
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

            # 顶栏：情绪 + 置信度（仅英文，OpenCV 不支持中文字体）
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
    ap = argparse.ArgumentParser(description="Real-time facial expression recognition")
    ap.add_argument("--ckpt", default="runs/classmate_model/best.pt")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--no-face-detect", action="store_true", help="skip face detection, classify full frame")
    ap.add_argument("--scale", type=float, default=1.0, help="display scale factor")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    model = load_model(a.ckpt, device)
    img_size = a.img_size
    print(f"[model] ResNet18 7-class  img_size={img_size}\n")

    # 加载 Haar 人脸检测器
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    use_face_detect = not a.no_face_detect

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print("[error] Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print(f"[camera] Started  mode={'face-detect' if use_face_detect else 'full-frame'}")
    print("  Q=Quit  S=Screenshot  F=Toggle face-detect\n")

    fps_times = []  # 用于平滑 FPS
    frame_count = 0
    tracker = FaceTracker()
    ff_probs_ema = None   # 整帧模式的 EMA 概率

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # 镜像
        display = frame.copy()
        faces_probs = []
        face_rects = []
        is_fullframe = False

        if use_face_detect:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # minNeighbors 提高到 7，减少眼睛/局部误检
            raw_faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=7,
                minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

            raw_rects, raw_probs_list = [], []
            if len(raw_faces) > 0:
                tensors = []
                for (x, y, fw, fh) in raw_faces:
                    tensors.append(preprocess_face(frame[y:y+fh, x:x+fw], img_size))
                    raw_rects.append((x, y, fw, fh))
                batch_x = torch.cat(tensors, dim=0)
                raw_probs_list = list(predict_batch(model, batch_x, device))

            confirmed = tracker.update(raw_rects, raw_probs_list)

            if confirmed:
                face_rects  = [r for r, _ in confirmed]
                faces_probs = [p for _, p in confirmed]
            else:
                # 没有已确认的人脸 → 回退整帧模式
                is_fullframe = True
                p = predict_batch(model, preprocess_face(frame, img_size), device)
                if len(p) > 0:
                    ff_probs_ema = (p[0].copy() if ff_probs_ema is None
                                    else _EMA_A * p[0] + (1 - _EMA_A) * ff_probs_ema)
                    faces_probs = [ff_probs_ema]
        else:
            is_fullframe = True
            p = predict_batch(model, preprocess_face(frame, img_size), device)
            if len(p) > 0:
                ff_probs_ema = (p[0].copy() if ff_probs_ema is None
                                else _EMA_A * p[0] + (1 - _EMA_A) * ff_probs_ema)
                faces_probs = [ff_probs_ema]

        # 绘制结果
        fps = 1.0 / max(time.time() - t0, 0.001)
        fps_times.append(fps)
        if len(fps_times) > 30:
            fps_times.pop(0)
        smooth_fps = np.mean(fps_times) if fps_times else 0

        display = draw_panel(display, faces_probs, face_rects, smooth_fps,
                             is_fullframe=is_fullframe)

        # 缩放
        if a.scale != 1.0:
            display = cv2.resize(display, None, fx=a.scale, fy=a.scale)

        cv2.imshow("FER Realtime — Emotion Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            out_path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(out_path, display)
            print(f"[screenshot] {out_path}")
        elif key == ord('f'):
            use_face_detect = not use_face_detect
            tracker.reset()
            ff_probs_ema = None
            print(f"[toggle] face-detect={'ON' if use_face_detect else 'OFF'}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[bye] Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
