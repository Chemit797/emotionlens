import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

EMO_COLOR = {
    "neutral":    (180, 180, 180),
    "happiness":  (0, 255, 100),
    "surprise":   (255, 200, 0),
    "sadness":    (200, 80, 0),
    "anger":      (0, 0, 240),
    "disgust":    (0, 140, 60),
    "fear":       (120, 0, 180),
}

_EMA_A      = 0.30
_MIN_HITS   = 3
_MAX_MISS   = 8
_IOU_THRESH = 0.25

def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
    iy = max(0, min(ay+ah, by+bh) - max(ay, by))
    inter = ix * iy
    union = aw*ah + bw*bh - inter
    return inter / union if union > 0 else 0.0

class FaceTracker:
    def __init__(self):
        self._tracks = []
        self._next_id = 1

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

        surviving = []
        for ti, t in enumerate(self._tracks):
            if ti not in used_trk:
                t['hits'] = 0
                t['miss'] += 1
            if t['miss'] <= _MAX_MISS:
                surviving.append(t)
        self._tracks = surviving

        for di, r in enumerate(raw_rects):
            if di not in used_det:
                self._tracks.append({
                    'id': self._next_id,
                    'rf': [float(v) for v in r],
                    'ri': list(r),
                    'probs': raw_probs_list[di].copy(),
                    'hits': 1, 'miss': 0, 'ok': False,
                })
                self._next_id += 1

        return [(t['id'], t['ri'], t['probs'])
                for t in self._tracks if t['ok'] and t['miss'] <= _MAX_MISS]

def load_model(ckpt_path, device, architecture="resnet18", state_key="model_state"):
    """Load a trained emotion classifier.

    Supported architectures:
      - resnet18        → torchvision ResNet-18 with custom fc (7-class)
      - efficientnet_b0 → timm EfficientNet-B0 (auto-adapts grayscale→RGB, 8→7 class)
      - effnet_tv       → torchvision EfficientNet-B0
    """
    ckpt_data = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt_data.get(state_key, ckpt_data)

    if architecture == "efficientnet_b0":
        import timm
        is_gray = sd.get("conv_stem.weight", torch.zeros(1)).shape[1] == 1
        in_chans = 1 if is_gray else 3
        num_ckpt_classes = sd.get("classifier.weight", torch.zeros(7, 1280)).shape[0]

        m = timm.create_model("efficientnet_b0", pretrained=False,
                              in_chans=in_chans, num_classes=num_ckpt_classes)
        m.load_state_dict(sd)

        # Adapt grayscale → RGB
        if is_gray:
            w = m.conv_stem.weight.data
            new_conv = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
            new_conv.weight.data = w.repeat(1, 3, 1, 1) / 3.0
            m.conv_stem = new_conv

        # Adapt 8-class (FER+) → 7-class
        if num_ckpt_classes == 8:
            old_cls = m.classifier
            new_cls = nn.Linear(old_cls.in_features, 7)
            new_cls.weight.data = old_cls.weight.data[:7, :]
            new_cls.bias.data = old_cls.bias.data[:7]
            m.classifier = new_cls
        elif num_ckpt_classes != 7:
            old_cls = m.classifier
            new_cls = nn.Linear(old_cls.in_features, 7)
            nn.init.xavier_uniform_(new_cls.weight)
            nn.init.zeros_(new_cls.bias)
            m.classifier = new_cls

        return m.to(device).eval()

    # ── Standard architectures (single load_state_dict) ──
    if architecture == "resnet18":
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 7)
    elif architecture == "effnet_tv":
        m = tvm.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 7)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    m.load_state_dict(sd)
    return m.to(device).eval()

def preprocess_face(face_bgr, img_size):
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
    if faces_tensor is None:
        return np.array([])
    x = faces_tensor.to(device)
    logits = model(x)
    return torch.softmax(logits, 1).cpu().numpy()

def draw_panel(frame, faces_probs, face_rects):
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
        cv2.putText(frame, label, (fx + 4, fy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

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

    return frame

class EngineFER:
    def __init__(self):
        from backend.config import MODEL_BACKEND, MODEL_REGISTRY

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Resolve which model to load
        if MODEL_BACKEND in MODEL_REGISTRY:
            cfg = MODEL_REGISTRY[MODEL_BACKEND]
            ckpt_path = os.path.abspath(cfg["path"])
            architecture = cfg["architecture"]
            state_key = cfg["state_key"]
            self.model_name = cfg["label"]
        else:
            # Treat MODEL_BACKEND as a filesystem path
            ckpt_path = os.path.abspath(MODEL_BACKEND)
            architecture = "resnet18"  # default assumption
            state_key = "model_state"
            self.model_name = os.path.basename(MODEL_BACKEND)

        print(f"[engine] Loading model: {self.model_name}")
        print(f"[engine]   path: {ckpt_path}")
        print(f"[engine]   arch: {architecture}")
        self.model = load_model(ckpt_path, self.device, architecture, state_key)
        self.img_size = 224

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.tracker = FaceTracker()

    def process_frame(self, frame, mode='m0', mode_state=None):
        # 1. Flip the frame JUST LIKE realtime_detect.py
        # This completely delegates the mirror logic to the backend.
        frame = cv2.flip(frame, 1)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        raw_faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=7,
            minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

        raw_rects, raw_probs_list = [], []
        if len(raw_faces) > 0:
            tensors = []
            for (x, y, fw, fh) in raw_faces:
                tensors.append(preprocess_face(frame[y:y+fh, x:x+fw], self.img_size))
                raw_rects.append((x, y, fw, fh))
            batch_x = torch.cat(tensors, dim=0)
            raw_probs_list = list(predict_batch(self.model, batch_x, self.device))

        confirmed = self.tracker.update(raw_rects, raw_probs_list)

        results = []
        face_rects = []
        faces_probs = []
        for tid, r_bbox, r_probs in confirmed:
            dom_idx = int(np.argmax(r_probs))
            dominant = EMO[dom_idx]
            
            probs_dict = {EMO[i]: float(r_probs[i]) for i in range(len(EMO))}
            results.append({
                'track_id': tid,
                'bbox': list(r_bbox),
                'conf': float(r_probs[dom_idx]),
                'dominant': dominant,
                'probs': probs_dict
            })
            face_rects.append(r_bbox)
            faces_probs.append(r_probs)
            
        # 2. Draw OpenCV UI
        rendered_frame = draw_panel(frame, faces_probs, face_rects)

        return results, rendered_frame
