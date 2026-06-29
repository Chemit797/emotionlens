"""Experiment 3: Data Ablation Study.

Compares ResNet18 trained on FERPlus-only vs ResNet18 trained on FERPlus+RAFDB+Self.
Isolates the impact of adding diverse real-world training data.

Requires: Exp2 (resnet18_ferplus_*) checkpoint + existing classmate_model/best.pt
"""
import json, argparse
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import cv2

EMO_7 = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
EMO_8 = EMO_7 + ["contempt"]

# -------------------------------------------------------------------
# Self data loading
# -------------------------------------------------------------------
def load_self_data(self_dir="self"):
    rows = []
    for user in sorted(Path(self_dir).iterdir()):
        if not user.is_dir() or user.name.startswith("."):
            continue
        pl = user / "prelabels.csv"
        if not pl.exists():
            continue
        df = pd.read_csv(pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        df["user"] = user.name
        rows.append(df)
    u2_pl = Path("user2_data/prelabels.csv")
    if u2_pl.exists():
        df = pd.read_csv(u2_pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        df["user"] = "user2"
        rows.append(df)
    all_df = pd.concat(rows, ignore_index=True)
    emo_map = {e: i for i, e in enumerate(EMO_7)}
    if "final_label" in all_df.columns:
        all_df["label"] = all_df["final_label"].map(emo_map)
        all_df = all_df[all_df["label"].notna()].reset_index(drop=True)
        all_df["label"] = all_df["label"].astype(int)
    return all_df

def resolve_image_path(filepath):
    fp = filepath.replace("\\", "/")
    if fp.startswith("raw/"):
        fp = fp[len("raw/"):]
    p = Path("self") / fp
    if p.exists():
        return p
    p2 = Path("user2_data") / fp
    if p2.exists():
        return p2
    return p

# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
def load_resnet18_ferplus(ckpt_path, device):
    """Load ResNet18 trained on FERPlus (8-class, RGB, state_key='model')."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model", ckpt)
    fc_weight = sd.get("fc.weight")
    num_classes = fc_weight.shape[0] if fc_weight is not None else 8
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    m.load_state_dict(sd)
    img_size = ckpt.get("img_size", ckpt.get("args", {}).get("img_size", 224))
    if isinstance(img_size, dict):
        img_size = 224
    return m.to(device).eval(), int(img_size)

def load_resnet18_classmate(ckpt_path, device):
    """Load classmate ResNet18 (7-class, RGB, state_key='model_state')."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state", ckpt)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(sd)
    img_size = ckpt.get("img_size", 224)
    return m.to(device).eval(), int(img_size)

# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
@torch.no_grad()
def infer_self_data(model, df, img_size, device, num_classes, bs=64):
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    all_preds = []
    for start in range(0, len(df), bs):
        batch = df.iloc[start:start+bs]
        batch_imgs = []
        for _, row in batch.iterrows():
            impath = resolve_image_path(row["filepath"])
            if impath.exists():
                try:
                    img = Image.open(impath).convert("RGB")
                    batch_imgs.append(preprocess(img))
                except Exception:
                    batch_imgs.append(torch.zeros(3, img_size, img_size))
            else:
                batch_imgs.append(torch.zeros(3, img_size, img_size))
        xb = torch.stack(batch_imgs).to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(xb)
        preds = out.argmax(1).cpu().numpy()
        # If model is 8-class, clip contempt(7) for 7-class evaluation
        if num_classes == 8:
            preds = np.clip(preds, 0, 6)
        all_preds.append(preds)
    return np.concatenate(all_preds)

@torch.no_grad()
def infer_ferplus_test(model, parquet_path, img_size, device, num_classes, bs=256):
    df = pd.read_parquet(parquet_path)
    te = df[df.split == "test"]
    pixels = te["pixels"].tolist()
    labels = te["label"].to_numpy().astype(np.int64)

    all_preds = []
    for start in range(0, len(pixels), bs):
        batch_pix = pixels[start:start+bs]
        batch_imgs = []
        for p_str in batch_pix:
            img_arr = np.array(p_str.split(), dtype=np.uint8).reshape(48, 48)
            img_arr = cv2.resize(img_arr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            img_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
            x = torch.from_numpy(img_rgb).float().permute(2,0,1).div_(255.0)
            x = (x - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            batch_imgs.append(x)
        xb = torch.stack(batch_imgs).to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(xb)
        preds = out.argmax(1).cpu().numpy()
        if num_classes == 8:
            preds = np.clip(preds, 0, 6)
        all_preds.append(preds)
    return np.concatenate(all_preds), labels

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ferplus-ckpt", default=None, help="ResNet18 FERPlus-only checkpoint")
    ap.add_argument("--classmate-ckpt", default="runs/classmate_model/best.pt")
    ap.add_argument("--ferplus-parquet", default="data/ferplus.parquet")
    ap.add_argument("--out-dir", default="runs/ablation")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Find Exp2 checkpoint if not specified
    if a.ferplus_ckpt is None:
        r18_dirs = sorted(Path("runs").glob("resnet18_ferplus_*"))
        if not r18_dirs:
            print("[ERROR] No ResNet18 FERPlus checkpoint found. Run Exp2 first.")
            return
        ferplus_ckpt = str(r18_dirs[-1] / "best.pt")
    else:
        ferplus_ckpt = a.ferplus_ckpt

    print(f"[checkpoints]")
    print(f"  FERPlus-only: {ferplus_ckpt}")
    print(f"  Classmate:    {a.classmate_ckpt}")

    # Load models
    model_ferplus, img_size_fp = load_resnet18_ferplus(ferplus_ckpt, device)
    model_cm, img_size_cm = load_resnet18_classmate(a.classmate_ckpt, device)
    print(f"  FERPlus-only ResNet18:  img_size={img_size_fp}, 8-class")
    print(f"  Classmate ResNet18:     img_size={img_size_cm}, 7-class")

    # Load test data
    self_df = load_self_data()
    self_labels = self_df["label"].values.astype(np.int64)
    print(f"\n[data] Self: {len(self_df)} images")

    # Evaluate both on self data
    print("\n[evaluating on self data]")
    self_preds_fp = infer_self_data(model_ferplus, self_df, img_size_fp, device, 8)
    self_preds_cm = infer_self_data(model_cm, self_df, img_size_cm, device, 7)

    acc_fp = (self_labels == self_preds_fp).mean()
    acc_cm = (self_labels == self_preds_cm).mean()
    f1_fp = f1_score(self_labels, self_preds_fp, average="macro", zero_division=0)
    f1_cm = f1_score(self_labels, self_preds_cm, average="macro", zero_division=0)

    print(f"  FERPlus-only:  acc={acc_fp:.4f}  f1={f1_fp:.4f}")
    print(f"  Classmate:     acc={acc_cm:.4f}  f1={f1_cm:.4f}")
    print(f"  Delta:         acc={acc_cm-acc_fp:+.4f}  f1={f1_cm-f1_fp:+.4f}")

    # Evaluate both on FERPlus test
    print("\n[evaluating on FERPlus test]")
    fp_test_preds_fp, fp_labels = infer_ferplus_test(model_ferplus, a.ferplus_parquet, img_size_fp, device, 8)
    fp_test_preds_cm, _ = infer_ferplus_test(model_cm, a.ferplus_parquet, img_size_cm, device, 7)

    # FERPlus test has 8 classes; for fair comparison only evaluate on 7 shared classes
    non_cont = fp_labels != 7
    fp_labels_7 = fp_labels[non_cont]
    fp_preds_fp_7 = fp_test_preds_fp[non_cont]
    fp_preds_cm_7 = fp_test_preds_cm[non_cont]

    acc_fp_test = (fp_labels_7 == fp_preds_fp_7).mean()
    acc_cm_test = (fp_labels_7 == fp_preds_cm_7).mean()
    f1_fp_test = f1_score(fp_labels_7, fp_preds_fp_7, average="macro", zero_division=0)
    f1_cm_test = f1_score(fp_labels_7, fp_preds_cm_7, average="macro", zero_division=0)

    print(f"  FERPlus-only:  acc={acc_fp_test:.4f}  f1={f1_fp_test:.4f}")
    print(f"  Classmate:     acc={acc_cm_test:.4f}  f1={f1_cm_test:.4f}")
    print(f"  Delta:         acc={acc_cm_test-acc_fp_test:+.4f}  f1={f1_cm_test-f1_fp_test:+.4f}")

    # ---- Per-class recall on self data ----
    per_class_fp = []
    per_class_cm = []
    for c in range(7):
        mask = self_labels == c
        per_class_fp.append(float((self_labels[mask] == self_preds_fp[mask]).mean()) if mask.sum() > 0 else float("nan"))
        per_class_cm.append(float((self_labels[mask] == self_preds_cm[mask]).mean()) if mask.sum() > 0 else float("nan"))

    # Print per-class comparison
    print(f"\n{'='*70}")
    print("Per-Class Recall on Self Data")
    print(f"{'='*70}")
    print(f"{'Emotion':>12s} | {'FERPlus-only':>12s} | {'Classmate':>12s} | {'Delta':>8s}")
    print("-" * 52)
    for i, emo in enumerate(EMO_7):
        d = per_class_cm[i] - per_class_fp[i] if not np.isnan(per_class_fp[i]) else float("nan")
        print(f"{emo:>12s} | {per_class_fp[i]:12.4f} | {per_class_cm[i]:12.4f} | {d:+8.4f}")

    # ---- Bar chart ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: overall comparison
    x = np.arange(2)
    w = 0.3
    datasets = ["Self Data", "FERPlus Test"]
    fp_accs = [acc_fp, acc_fp_test]
    cm_accs = [acc_cm, acc_cm_test]
    fp_f1s = [f1_fp, f1_fp_test]
    cm_f1s = [f1_cm, f1_cm_test]

    ax1.bar(x - w/2, fp_accs, w, label="FERPlus-only (Acc)", color="#3498db", edgecolor="white")
    ax1.bar(x + w/2, cm_accs, w, label="F+R+S Classmate (Acc)", color="#e74c3c", edgecolor="white")
    ax1.bar(x - w/2, fp_f1s, w, label="FERPlus-only (F1)", color="#85c1e9", edgecolor="white", alpha=0.6)
    ax1.bar(x + w/2, cm_f1s, w, label="F+R+S Classmate (F1)", color="#f1948a", edgecolor="white", alpha=0.6)
    for i, (va, vb) in enumerate(zip(fp_accs, cm_accs)):
        ax1.text(i - w/2, va + 0.015, f"{va:.3f}", ha="center", fontsize=9, fontweight="bold")
        ax1.text(i + w/2, vb + 0.015, f"{vb:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Overall Accuracy & F1 Comparison")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(axis="y", alpha=0.3)

    # Right: per-class delta
    x2 = np.arange(7)
    deltas = [per_class_cm[i] - per_class_fp[i] if not np.isnan(per_class_fp[i]) else 0 for i in range(7)]
    colors = ["#27ae60" if d > 0 else "#e74c3c" if d < 0 else "#95a5a6" for d in deltas]
    bars = ax2.bar(x2, deltas, color=colors, edgecolor="white")
    for bar, d in zip(bars, deltas):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if d >= 0 else -0.06),
                f"{d:+.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(EMO_7, rotation=45, ha="right")
    ax2.set_ylabel("Delta Recall (Classmate - FERPlus-only)")
    ax2.set_title("Per-Class Improvement from Adding RAFDB+Self")
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Data Ablation: FERPlus-only vs FERPlus+RAFDB+Self (ResNet18)\n"
                 f"Self Data: FERPlus={acc_fp:.3f} -> Classmate={acc_cm:.3f}  "
                 f"FERPlus Test: {acc_fp_test:.3f} -> {acc_cm_test:.3f}",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(out / "ablation_comparison.png", dpi=150)
    plt.close()
    print(f"\n[saved] {out / 'ablation_comparison.png'}")

    # Save JSON
    result = {
        "ferplus_only": {
            "self_acc": float(acc_fp), "self_f1": float(f1_fp),
            "ferplus_test_acc": float(acc_fp_test), "ferplus_test_f1": float(f1_fp_test),
            "per_class_recall_self": per_class_fp,
        },
        "classmate_frs": {
            "self_acc": float(acc_cm), "self_f1": float(f1_cm),
            "ferplus_test_acc": float(acc_cm_test), "ferplus_test_f1": float(f1_cm_test),
            "per_class_recall_self": per_class_cm,
        },
        "delta": {
            "self_acc": float(acc_cm - acc_fp),
            "self_f1": float(f1_cm - f1_fp),
            "ferplus_test_acc": float(acc_cm_test - acc_fp_test),
            "ferplus_test_f1": float(f1_cm_test - f1_fp_test),
            "per_class_delta": deltas,
        },
    }
    json.dump(result, open(out / "ablation_results.json", "w"), indent=2)
    print(f"[saved] {out / 'ablation_results.json'}")

    print(f"\n[EXP3 DONE] Results in {out}/")


if __name__ == "__main__":
    main()
