"""命令行表情识别 —— 输入图片路径，输出预测结果"""
import argparse
from pathlib import Path
import numpy as np
import torch
import timm
import cv2

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
EMO_CN = {"neutral": "平静", "happiness": "开心", "surprise": "惊讶", "sadness": "悲伤",
          "anger": "愤怒", "disgust": "厌恶", "fear": "恐惧", "contempt": "轻蔑"}


def load_model(ckpt_path, model_name, device):
    m = timm.create_model(model_name, pretrained=False, num_classes=8, in_chans=1)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return m.to(device).eval()


def preprocess(image_path, img_size):
    """加载图片 → 灰度 → 48×48 → img_size×img_size → 归一化"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 先缩放到 48×48（模拟 FER2013 原始分辨率）
    face = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
    # 再缩放到模型输入尺寸
    face = cv2.resize(face, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # 归一化（与训练一致）
    x = torch.from_numpy(face).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0).unsqueeze(0)
    return x, gray  # 返回 tensor 和原始灰度图


@torch.no_grad()
def predict(model, image_path, img_size, device, topk=3):
    x, gray = preprocess(image_path, img_size)
    x = x.to(device)
    logits = model(x)
    probs = torch.softmax(logits, 1).cpu().numpy()[0]
    order = np.argsort(probs)[::-1]
    return [(EMO[i], EMO_CN[EMO[i]], float(probs[i])) for i in order[:topk]], gray


def main():
    ap = argparse.ArgumentParser(description="FER 表情识别")
    ap.add_argument("image", nargs="+", help="图片路径（可多个）")
    ap.add_argument("--ckpt", default="runs/efficientnet_b0_20260615_235918/best.pt")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--topk", type=int, default=3, help="显示 top-K 预测")
    ap.add_argument("--save", action="store_true", help="保存标注后的图片")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_data = torch.load(a.ckpt, map_location=device)
    img_size = ckpt_data.get("args", {}).get("img_size", a.img_size)
    model = load_model(a.ckpt, a.model, device)
    print(f"[model] {a.model}  img_size={img_size}  device={device}\n")

    for impath in a.image:
        path = Path(impath)
        if not path.exists():
            print(f"[skip] not found: {impath}")
            continue
        try:
            results, gray = predict(model, str(path), img_size, device, a.topk)
        except Exception as e:
            print(f"[error] {impath}: {e}")
            continue

        print(f"=== {path.name} ===")
        for rank, (emo, cn, conf) in enumerate(results):
            bar = "█" * int(conf * 40)
            print(f"  {rank+1}. {emo:>10s} ({cn})  {conf:.1%}  {bar}")

        if a.save:
            # 在灰度图上标注
            annotated = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            annotated = cv2.resize(annotated, (480, 480))
            top_emo, top_cn, top_conf = results[0]
            cv2.putText(annotated, f"{top_emo} ({top_cn}) {top_conf:.1%}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            for rank, (emo, cn, conf) in enumerate(results[1:], 1):
                cv2.putText(annotated, f"{emo} ({cn}) {conf:.1%}",
                            (10, 40 + rank * 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (200, 200, 200), 1)
            out_path = path.parent / f"{path.stem}_pred{path.suffix}"
            cv2.imwrite(str(out_path), annotated)
            print(f"  -> saved: {out_path}")
        print()


if __name__ == "__main__":
    main()
