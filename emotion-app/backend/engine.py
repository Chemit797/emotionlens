import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
from backend.config import EMO_CLASSES, DET_CONF, MIN_FACE_PX, EMA_GAMMA

device = "cuda" if torch.cuda.is_available() else "cpu"

class Engine:
    def __init__(self, model_path="models/best.pt", yunet_path="models/face_detection_yunet.onnx"):
        # Load YuNet
        self.detector = cv2.FaceDetectorYN.create(
            yunet_path, "", (320, 320), score_threshold=DET_CONF
        )
        
        # Load ResNet18 Model
        ckpt = torch.load(model_path, map_location=device)
        m = tvm.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 7)
        m.load_state_dict(ckpt["model_state"])
        self.model = m.to(device).eval()
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        self.tracks = {}
        self.next_track_id = 1
        
    def _iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ix = max(0, min(ax+aw, bx+bw) - max(ax, bx))
        iy = max(0, min(ay+ah, by+bh) - max(ay, by))
        inter = ix * iy
        union = aw*ah + bw*bh - inter
        return inter / union if union > 0 else 0.0

    def process_frame(self, bgr_img):
        h, w = bgr_img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_img)
        
        results = []
        current_tracks = {}
        
        if faces is not None:
            raw_rects = []
            for face in faces:
                box = face[0:4].astype(np.int32)
                conf = face[-1]
                x, y, fw, fh = box
                if min(fw, fh) < MIN_FACE_PX:
                    continue
                # Ensure box inside image
                x1 = max(0, x)
                y1 = max(0, y)
                x2 = min(w, x + fw)
                y2 = min(h, y + fh)
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Model predict
                face_img = bgr_img[y1:y2, x1:x2]
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb)
                tensor = self.preprocess(face_pil).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                
                raw_rects.append({'bbox': box.tolist(), 'probs': probs, 'conf': float(conf)})

            # Tracker
            used_det = set()
            for tid, t in self.tracks.items():
                best_iou = 0.25
                best_det_idx = -1
                for i, r in enumerate(raw_rects):
                    if i in used_det:
                        continue
                    iou = self._iou(t['bbox'], r['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_det_idx = i
                if best_det_idx >= 0:
                    used_det.add(best_det_idx)
                    r = raw_rects[best_det_idx]
                    # EMA smoothing
                    smoothed_probs = EMA_GAMMA * r['probs'] + (1 - EMA_GAMMA) * t['probs']
                    # Smoothing bbox too
                    s_bbox = [int(EMA_GAMMA * r['bbox'][k] + (1 - EMA_GAMMA) * t['bbox'][k]) for k in range(4)]
                    
                    dom_idx = int(np.argmax(smoothed_probs))
                    
                    current_tracks[tid] = {
                        'track_id': tid,
                        'bbox': s_bbox,
                        'probs': smoothed_probs,
                        'conf': r['conf'],
                        'dominant': EMO_CLASSES[dom_idx]
                    }
                    
            for i, r in enumerate(raw_rects):
                if i not in used_det:
                    tid = self.next_track_id
                    self.next_track_id += 1
                    dom_idx = int(np.argmax(r['probs']))
                    current_tracks[tid] = {
                        'track_id': tid,
                        'bbox': r['bbox'],
                        'probs': r['probs'],
                        'conf': r['conf'],
                        'dominant': EMO_CLASSES[dom_idx]
                    }
                    
        self.tracks = current_tracks
        
        # Prepare final output
        for t in current_tracks.values():
            probs_dict = {EMO_CLASSES[i]: float(t['probs'][i]) for i in range(len(EMO_CLASSES))}
            results.append({
                'track_id': t['track_id'],
                'bbox': t['bbox'],
                'conf': t['conf'],
                'dominant': t['dominant'],
                'probs': probs_dict
            })
            
        return results
