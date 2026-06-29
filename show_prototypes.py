"""展示每类表情的标准案例（模型最有信心的样本）"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import timm
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]


def load_model(ckpt_path, model_name, device):
    m = timm.create_model(model_name, pretrained=False, num_classes=8, in_chans=1)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return m.to(device).eval()


@torch.no_grad()
def get_predictions(model, pixels, img_size, device):
    """对像素列表做推理，返回 softmax 概率"""
    probs = []
    batch_size = 256
    for start in range(0, len(pixels), batch_size):
        batch_pix = pixels[start:start + batch_size]
        batch_imgs = []
        for p in batch_pix:
            img = np.array(p.split(), dtype=np.uint8).reshape(48, 48)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            x = torch.from_numpy(img).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
            batch_imgs.append(x)
        x_batch = torch.stack(batch_imgs).to(device)
        out = model(x_batch)
        probs.append(torch.softmax(out, 1).cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/efficientnet_b0_20260615_235918/best.pt")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--data", default="data/ferplus.parquet")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--top-k", type=int, default=5, help="每类展示几张")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # 加载模型
    model = load_model(a.ckpt, a.model, device)
    ckpt_args = torch.load(a.ckpt, map_location=device).get("args", {})
    img_size = ckpt_args.get("img_size", a.img_size)
    print(f"[model] {a.model}  img_size={img_size}")

    # 加载测试集
    df = pd.read_parquet(a.data)
    te = df[df.split == "test"].reset_index(drop=True)
    print(f"[data] test set: {len(te)} images")

    # 推理
    print("[inference] running...")
    probs = get_predictions(model, te["pixels"].tolist(), img_size, device)
    preds = probs.argmax(1)

    # 对每类，按"预测为该类"的概率排序，取 top-K
    fig, axes = plt.subplots(8, a.top_k, figsize=(a.top_k * 2, 16))
    for cls_idx in range(8):
        # 找出预测为该类的样本，按该类概率降序
        mask = preds == cls_idx
        cls_probs = probs[mask, cls_idx]
        idx_in_mask = np.argsort(cls_probs)[::-1][:a.top_k]
        # 在原 df 中的位置
        orig_indices = np.where(mask)[0][idx_in_mask]

        for k, orig_idx in enumerate(orig_indices):
            ax = axes[cls_idx, k]
            img = np.array(te.iloc[orig_idx]["pixels"].split(), dtype=np.uint8).reshape(48, 48)
            ax.imshow(img, cmap="gray")
            conf = probs[orig_idx, cls_idx]
            true_label = te.iloc[orig_idx]["label"]
            true_emo = EMO[true_label]
            # 如果预测和真实标签不同，用红色标出
            color = "green" if true_label == cls_idx else "red"
            ax.set_title(f"{conf:.2f} | true:{true_emo}", fontsize=8, color=color)
            ax.axis("off")
            if k == 0:
                ax.set_ylabel(EMO[cls_idx], fontsize=12, fontweight="bold", rotation=0,
                              labelpad=40, va="center")

    plt.suptitle(f"Top-{a.top_k} Prototypical Examples per Emotion (green=correct, red=mislabeled)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = "runs/prototypes.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[done] saved {out_path}")

    # 打印摘要
    print("\n=== Per-class summary ===")
    for cls_idx in range(8):
        mask = preds == cls_idx
        correct = (preds[mask] == te.iloc[np.where(mask)[0]]["label"].values).mean()
        mean_conf = probs[mask, cls_idx].mean()
        print(f"  {EMO[cls_idx]:>10s}: predicted {mask.sum():>5d} times, "
              f"correct {correct:.1%}, mean confidence {mean_conf:.3f}")


if __name__ == "__main__":
    main()
