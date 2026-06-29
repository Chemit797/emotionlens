"""
EmotionLens Demo Inference Script
==================================
Runs the trained ResNet18 model on two test sets:
  1. self_test/  -- our own faces (6 team members)
  2. fer_test/   -- FERPlus benchmark faces

Usage:
    python run_demo.py                    # run on both test sets
    python run_demo.py --device cuda      # use GPU if available
    python run_demo.py --self-only        # only self_test
    python run_demo.py --fer-only         # only fer_test
    python run_demo.py --image face.jpg   # single image inference

Requires: torch, torchvision, pillow, numpy, pandas
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
EMO_CN = {
    "neutral": "neutral",
    "happiness": "happiness",
    "surprise": "surprise",
    "sadness": "sadness",
    "anger": "anger",
    "disgust": "disgust",
    "fear": "fear",
}

BASE_DIR = Path(__file__).resolve().parent
IMG_SIZE = 224
PREPROCESS = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(ckpt_path, device):
    """Load ResNet18 7-class model."""
    print(f"[model] Loading {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(ckpt["model_state"])
    m = m.to(device).eval()
    n = sum(p.numel() for p in m.parameters())
    print(f"[model] ResNet18 7-class | {n:,} params | device={device}")
    return m


def load_image(path):
    """Load and preprocess one image."""
    img = Image.open(path).convert("RGB")
    return PREPROCESS(img).unsqueeze(0)


@torch.no_grad()
def predict_batch(model, paths, device, bs=32):
    """Batch inference on image paths. Returns (probs, preds)."""
    all_probs = []
    for start in range(0, len(paths), bs):
        batch = paths[start:start + bs]
        tensors = []
        for p in batch:
            try:
                tensors.append(load_image(p))
            except Exception as e:
                print(f"  [WARN] {Path(p).name}: {e}")
                tensors.append(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
        xb = torch.cat(tensors).to(device)
        probs = torch.softmax(m(xb), dim=1).cpu().numpy()
        all_probs.append(probs)
    if not all_probs:
        return np.array([]), np.array([])
    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(axis=1)
    return all_probs, all_preds


def print_bar(probs, pred_idx, true_idx=None):
    """Horizontal probability bar chart."""
    w = 40
    for i, emo in enumerate(EMO):
        p = probs[i]
        filled = int(p * w)
        bar = "#" * filled + "-" * (w - filled)
        tags = []
        if true_idx is not None and i == true_idx:
            tags.append("TRUE")
        if i == pred_idx:
            tags.append("PRED")
        tag = " (" + ",".join(tags) + ")" if tags else ""
        print(f"  {emo:>10s} |{bar}| {p:.3f}{tag}")


def header(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ================================================================
# Self Test
# ================================================================
def eval_self(model, device):
    """Evaluate on self_test/ with ground-truth from labels.csv."""
    lp = BASE_DIR / "labels.csv"
    if not lp.exists():
        print("[SKIP] labels.csv not found")
        return None

    df = pd.read_csv(lp)
    img_dir = BASE_DIR / "self_test"

    header("SELF TEST -- Our Own Faces (6 team members)")
    print(f"  Images: {len(df)} | Users: {df['user'].nunique()} | "
          f"Emotions: {df['true_emotion'].nunique()}")

    # Build path list, tracking missing
    paths = []
    for _, row in df.iterrows():
        p = img_dir / row["filename"]
        paths.append(p if p.exists() else None)

    valid_idx = [i for i, p in enumerate(paths) if p is not None]
    valid_paths = [paths[i] for i in valid_idx]

    t0 = time.time()
    probs, preds = predict_batch(model, valid_paths, device)
    elapsed = time.time() - t0

    # Map back to full dataframe
    all_p = np.full(len(df), -1)
    all_prob = np.zeros((len(df), 7))
    for j, idx in enumerate(valid_idx):
        all_p[idx] = preds[j]
        all_prob[idx] = probs[j]

    tl = df["label_idx"].values
    vm = all_p >= 0
    correct = int((all_p[vm] == tl[vm]).sum())
    total = int(vm.sum())
    acc = correct / total if total > 0 else 0

    df["pred"] = all_p
    df["correct"] = (df["pred"] == df["label_idx"]).astype(int)

    print(f"\n  Overall Accuracy: {correct}/{total} = {acc:.1%}")
    print(f"  Time: {elapsed:.2f}s ({elapsed/total*1000:.1f} ms/img)\n")

    # Per-user table
    print(f"  {'User':>8s} | {'Imgs':>5s} | {'Correct':>8s} | {'Acc':>7s}")
    print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*7}")
    for user in sorted(df["user"].unique()):
        u = df[df["user"] == user]
        c = int(u["correct"].sum())
        t = len(u)
        print(f"  {user:>8s} | {t:>5d} | {c:>5d}/{t:<4d} | {c/t:>6.1%}")

    # Per-emotion recall
    print(f"\n  {'Emotion':>10s} | {'Recall':>8s} | {'N':>5s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}")
    for i, emo in enumerate(EMO):
        mask = tl == i
        if mask.sum() > 0:
            r = (all_p[mask] == i).sum()
            print(f"  {emo:>10s} | {r/int(mask.sum()):>7.1%} | {int(mask.sum()):>4d}")

    # Sample predictions (one per user)
    print(f"\n  --- Sample Predictions (one per user) ---")
    for user in sorted(df["user"].unique()):
        uidx = df[df["user"] == user].index[0]
        row = df.loc[uidx]
        pi = int(row["pred"])
        ti = int(row["label_idx"])
        prob_row = all_prob[uidx]
        ok = "OK" if pi == ti else "XX"
        print(f"\n  [{ok}] {row['filename']}")
        print(f"       True: {EMO[ti]}  -->  Pred: {EMO[pi]}  "
              f"(conf={prob_row[pi]:.3f})")

    return {"accuracy": acc, "total": total, "correct": correct}


# ================================================================
# FER Test
# ================================================================
def eval_fer(model, device):
    """Evaluate on fer_test/ images (named ferplus_<emotion>_<n>.png)."""
    img_dir = BASE_DIR / "fer_test"
    if not img_dir.exists():
        print("[SKIP] fer_test/ not found")
        return None

    paths = sorted(img_dir.glob("*.png"))
    if not paths:
        print("[SKIP] fer_test/ is empty")
        return None

    # Parse labels from filename: ferplus_neutral_1.png -> neutral
    tl = []
    for p in paths:
        parts = p.stem.split("_")
        emo = parts[1] if len(parts) >= 2 else "unknown"
        tl.append(EMO.index(emo) if emo in EMO else -1)

    header("FER TEST -- FERPlus Benchmark Faces")
    print(f"  Images: {len(paths)} | Emotions: 7")

    t0 = time.time()
    probs, preds = predict_batch(model, paths, device)
    elapsed = time.time() - t0

    ta = np.array(tl)
    correct = int((preds == ta).sum())
    total = len(preds)
    acc = correct / total if total > 0 else 0

    print(f"\n  Overall Accuracy: {correct}/{total} = {acc:.1%}")
    print(f"  Time: {elapsed:.2f}s ({elapsed/total*1000:.1f} ms/img)\n")

    # Per-emotion recall
    print(f"  {'Emotion':>10s} | {'Recall':>8s} | {'N':>5s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*5}")
    for i, emo in enumerate(EMO):
        mask = ta == i
        if mask.sum() > 0:
            r = (preds[mask] == i).sum()
            print(f"  {emo:>10s} | {r/int(mask.sum()):>7.1%} | {int(mask.sum()):>4d}")

    # Detailed predictions with bar charts
    print(f"\n  --- Detailed Predictions ---")
    for i, p in enumerate(paths):
        ti = ta[i]
        pi = int(preds[i])
        ok = "OK" if pi == ti else "XX"
        print(f"\n  [{ok}] {p.name}")
        print(f"       True: {EMO[ti]}  -->  Pred: {EMO[pi]}  "
              f"(conf={probs[i][pi]:.3f})")
        print_bar(probs[i], pi, ti)

    return {"accuracy": acc, "total": total, "correct": correct}


# ================================================================
# Single Image Mode
# ================================================================
def infer_single(model, device, img_paths):
    """Run inference on arbitrary images."""
    header("SINGLE IMAGE INFERENCE")
    for img_path in img_paths:
        p = Path(img_path)
        if not p.exists():
            print(f"  [SKIP] Not found: {img_path}")
            continue
        probs, preds = predict_batch(model, [p], device)
        pi = int(preds[0])
        print(f"\n  {p.name}")
        print(f"  Predicted: {EMO[pi]}  conf={probs[0][pi]:.3f}")
        print_bar(probs[0], pi)


# ================================================================
# Main
# ================================================================
def main():
    ap = argparse.ArgumentParser(description="EmotionLens Demo Inference")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Device to use for inference")
    ap.add_argument("--ckpt", default=None,
                    help="Path to model checkpoint (default: demo_kit/best.pt)")
    ap.add_argument("--self-only", action="store_true",
                    help="Only evaluate self_test")
    ap.add_argument("--fer-only", action="store_true",
                    help="Only evaluate fer_test")
    ap.add_argument("--image", type=str, nargs="*",
                    help="Run inference on specific image(s)")
    args = ap.parse_args()

    # Device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        device = "cpu"

    # Model
    ckpt = Path(args.ckpt) if args.ckpt else BASE_DIR / "best.pt"
    if not ckpt.exists():
        print(f"[ERROR] Model not found: {ckpt}")
        print("  Place best.pt in demo_kit/ or use --ckpt <path>")
        sys.exit(1)
    model = load_model(ckpt, device)

    # Single image mode
    if args.image:
        infer_single(model, device, args.image)
        return

    # Test set evaluation
    results = {}
    if not args.fer_only:
        results["self"] = eval_self(model, device)
    if not args.self_only:
        results["fer"] = eval_fer(model, device)

    # Summary
    header("SUMMARY")
    if results.get("self"):
        s = results["self"]
        print(f"  Self Test Accuracy:  {s['accuracy']:.1%}  "
              f"({s['correct']}/{s['total']})")
    if results.get("fer"):
        f = results["fer"]
        print(f"  FER Test Accuracy:   {f['accuracy']:.1%}  "
              f"({f['correct']}/{f['total']})")

    if results.get("self") and results.get("fer"):
        gap = results["fer"]["accuracy"] - results["self"]["accuracy"]
        print(f"  Domain Gap (Delta):  {gap:+.1%}")
        print()
        if gap > 0.10:
            print("  The model performs much better on FERPlus (lab faces) than")
            print("  on our real webcam faces. This 'domain gap' is exactly what")
            print("  our report investigates.")
        print()
        print("  Full reports:")
        print("    report_v2_en_v3.tex  (English)")
        print("    report_v1_cn_v3.tex  (Chinese / 中文)")

    # Save results.txt
    with open(BASE_DIR / "results.txt", "w", encoding="utf-8") as out:
        out.write("EmotionLens Demo Inference Results\n")
        out.write("=" * 50 + "\n\n")
        if results.get("self"):
            s = results["self"]
            out.write(f"Self Test Accuracy: {s['accuracy']:.1%} "
                      f"({s['correct']}/{s['total']})\n")
        if results.get("fer"):
            f = results["fer"]
            out.write(f"FER Test Accuracy:  {f['accuracy']:.1%} "
                      f"({f['correct']}/{f['total']})\n")
        if results.get("self") and results.get("fer"):
            out.write(f"Domain Gap (Delta): {gap:+.1%}\n")

    print(f"\n  Results also saved to results.txt")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
