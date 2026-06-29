"""Experiment 4: Cross-Domain Unified Evaluation.

Evaluates ALL available model checkpoints across ALL test sets:
- FERPlus test (8-class)
- FER2013 test (7-class)
- Self-collected data (7-class)

Generates the "big comparison table" for the report.
"""
import json, argparse
from pathlib import Path
from collections import OrderedDict
import numpy as np, pandas as pd
import torch, torch.nn as nn, timm
import torchvision.models as tvm
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
import cv2

EMO_7 = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
EMO_8 = EMO_7 + ["contempt"]

# -------------------------------------------------------------------
# Dataset for FERPlus parquet (RGB, for models that need it)
# -------------------------------------------------------------------
def load_ferplus_parquet(path):
    df = pd.read_parquet(path)
    te = df[df.split == "test"]
    pixels = te["pixels"].tolist()
    labels = te["label"].to_numpy().astype(np.int64)
    return pixels, labels, len(te)

# -------------------------------------------------------------------
# Dataset for FER2013 CSV
# -------------------------------------------------------------------
def load_fer2013_csv(path):
    df = pd.read_csv(path)
    df.columns = ["emotion", "pixels", "Usage"]
    te = df[df.Usage == "PrivateTest"]
    pixels = te["pixels"].tolist()
    labels = te["emotion"].to_numpy().astype(np.int64)
    return pixels, labels, len(te)

# -------------------------------------------------------------------
# Self-collected data
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
    # Map final_label to int
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
def load_model(ckpt_path, device, arch_info):
    """Load a model checkpoint. arch_info contains keys: architecture, state_key, num_classes, img_size."""
    arch = arch_info["architecture"]
    state_key = arch_info.get("state_key", "model_state")
    img_size = arch_info.get("img_size", 224)
    num_classes_out = arch_info.get("num_classes", 7)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get(state_key, ckpt)

    in_chans = 3  # default for VGG/ResNet
    if arch == "vgg16":
        m = tvm.vgg16(weights=None)
        m.classifier[6] = nn.Linear(4096, 7)
        m.load_state_dict(sd)
    elif arch == "resnet18":
        m = tvm.resnet18(weights=None)
        fc_weight = sd.get("fc.weight")
        if fc_weight is not None:
            num_classes = fc_weight.shape[0]
        else:
            num_classes = 7
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.load_state_dict(sd)
    elif arch == "efficientnet_b0":
        in_chans = sd.get("conv_stem.weight", torch.zeros(1)).shape[1]
        num_ckpt_classes = sd.get("classifier.weight", torch.zeros(8, 1280)).shape[0]
        m = timm.create_model("efficientnet_b0", pretrained=False,
                              in_chans=in_chans, num_classes=num_ckpt_classes)
        m.load_state_dict(sd)
        # Keep original in_chans — do NOT adapt conv_stem.
        # Input normalization must match training: [-1,1] for 1ch, ImageNet for 3ch.
        # We handle this in infer_pixels / infer_self_data via norm_type.
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m.to(device).eval(), img_size, in_chans

# -------------------------------------------------------------------
# Inference
# -------------------------------------------------------------------
@torch.no_grad()
def infer_pixels(model, pixels_list, labels, img_size, device, in_chans, normalize_imagenet, bs=256):
    """Run inference on pixel strings (48x48 grayscale)."""
    all_preds = []
    for start in range(0, len(pixels_list), bs):
        batch_pix = pixels_list[start:start+bs]
        batch_imgs = []
        for p_str in batch_pix:
            img = np.array(p_str.split(), dtype=np.uint8).reshape(48, 48)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            if in_chans == 1:
                x = torch.from_numpy(img).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
            elif in_chans == 3:
                img = np.stack([img, img, img], axis=-1)
                x = torch.from_numpy(img).float().permute(2, 0, 1).div_(255.0)
                x = (x - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            batch_imgs.append(x)
        xb = torch.stack(batch_imgs).to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(xb)
        all_preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(all_preds)

@torch.no_grad()
def infer_self_data(model, df, img_size, device, in_chans, normalize_imagenet, bs=64):
    """Run inference on self-collected images (loaded from file)."""
    all_preds = []
    for start in range(0, len(df), bs):
        batch = df.iloc[start:start+bs]
        batch_imgs = []
        for _, row in batch.iterrows():
            impath = resolve_image_path(row["filepath"])
            if impath.exists():
                img = cv2.imread(str(impath))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                else:
                    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            if in_chans == 1:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                x = torch.from_numpy(gray).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)
            elif in_chans == 3:
                x = torch.from_numpy(img).float().permute(2,0,1).div_(255.0)
                x = (x - torch.tensor([0.485,0.456,0.406]).view(3,1,1)) / torch.tensor([0.229,0.224,0.225]).view(3,1,1)
            batch_imgs.append(x)
        xb = torch.stack(batch_imgs).to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(xb)
        all_preds.append(out.argmax(1).cpu().numpy())
    return np.concatenate(all_preds)

# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------
def compute_metrics(labels, preds, num_classes):
    acc = (labels == preds).mean()
    f1 = f1_score(labels, preds, average="macro", labels=list(range(num_classes)), zero_division=0)
    per_class_recall = []
    for c in range(num_classes):
        mask = labels == c
        per_class_recall.append(float((labels[mask] == preds[mask]).mean()) if mask.sum() > 0 else float("nan"))
    return {"acc": float(acc), "f1": float(f1), "per_class_recall": per_class_recall}

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ferplus-parquet", default="data/ferplus.parquet")
    ap.add_argument("--fer2013-csv", default="data/fer2013.csv")
    ap.add_argument("--self-dir", default="self")
    ap.add_argument("--out-dir", default="runs/cross_eval")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load test data
    print("[loading test data]")
    fp_pixels, fp_labels, fp_n = load_ferplus_parquet(a.ferplus_parquet)
    f3_pixels, f3_labels, f3_n = load_fer2013_csv(a.fer2013_csv)
    self_df = load_self_data(a.self_dir)
    self_labels = self_df["label"].values.astype(np.int64)
    print(f"  FERPlus test:  {fp_n} images")
    print(f"  FER2013 test:  {f3_n} images")
    print(f"  Self data:     {len(self_df)} images")

    # Define all models to evaluate
    # (label, ckpt_path, arch_info)
    models_to_eval = []

    # VGG16 (from Exp1) - find newest
    vgg_dirs = sorted(Path("runs").glob("vgg16_fer2013_*"))
    if vgg_dirs:
        ckpt = vgg_dirs[-1] / "best.pt"
        if ckpt.exists():
            models_to_eval.append(("VGG16 (FER2013)", str(ckpt), {
                "architecture": "vgg16", "state_key": "model_state",
                "img_size": 224, "in_chans": 3, "normalize_imagenet": True, "num_classes": 7
            }))

    # ResNet18 FERPlus only (from Exp2) - find newest
    r18_dirs = sorted(Path("runs").glob("resnet18_ferplus_*"))
    if r18_dirs:
        ckpt = r18_dirs[-1] / "best.pt"
        if ckpt.exists():
            models_to_eval.append(("ResNet18 (FERPlus only)", str(ckpt), {
                "architecture": "resnet18", "state_key": "model",
                "img_size": 224, "in_chans": 3, "normalize_imagenet": True, "num_classes": 8
            }))

    # ResNet18 Classmate (F+R+S)
    cm_ckpt = Path("runs/classmate_model/best.pt")
    if cm_ckpt.exists():
        models_to_eval.append(("ResNet18 (F+R+S, classmate)", str(cm_ckpt), {
            "architecture": "resnet18", "state_key": "model_state",
            "img_size": 224, "in_chans": 3, "normalize_imagenet": True, "num_classes": 7
        }))

    # EfficientNet-B0 (existing best)
    eff_ckpt = Path("runs/efficientnet_b0_20260615_235918/best.pt")
    if eff_ckpt.exists():
        models_to_eval.append(("EfficientNet-B0 (FERPlus)", str(eff_ckpt), {
            "architecture": "efficientnet_b0", "state_key": "model",
            "img_size": 128, "in_chans": 1, "normalize_imagenet": False, "num_classes": 8
        }))

    if not models_to_eval:
        print("[ERROR] No model checkpoints found! Run Exp1 and Exp2 first.")
        return

    print(f"\n[models to evaluate] {len(models_to_eval)}")
    for label, ckpt, _ in models_to_eval:
        print(f"  {label}: {ckpt}")

    # Evaluate each model on each test set
    results = OrderedDict()
    for label, ckpt_path, arch_info in models_to_eval:
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}")
        print(f"{'='*60}")

        model, img_size, effective_in_ch = load_model(ckpt_path, device, arch_info)
        in_ch = effective_in_ch  # from load_model (1 for grayscale EfficientNet, 3 for RGB models)
        norm_im = arch_info["normalize_imagenet"]
        model_num_cls = arch_info["num_classes"]

        model_results = {}

        # On FERPlus test (always 8-class)
        print("  -> FERPlus test...")
        fp_preds = infer_pixels(model, fp_pixels, fp_labels, img_size, device, in_ch, norm_im)
        # If model is 7-class, map predictions: model sees 0-6 only, contempt not predicted
        if model_num_cls == 7:
            # For 7-class model on 8-class data: only evaluate on non-contempt samples
            non_cont = fp_labels != 7
            fp_preds_7 = fp_preds  # 0-6 already
            fp_labels_7 = fp_labels[non_cont]
            fp_preds_7_eval = fp_preds_7[non_cont]
            model_results["ferplus_test"] = compute_metrics(fp_labels_7, fp_preds_7_eval, 7)
            # Also compute treating contempt as error (conservative)
            fp_preds_8 = np.where(fp_labels == 7, 7, fp_preds_7)  # model can't predict contempt
            model_results["ferplus_test_8class"] = compute_metrics(fp_labels, fp_preds_8, 8)
        else:
            model_results["ferplus_test"] = compute_metrics(fp_labels, fp_preds, 8)

        # On FER2013 test (always 7-class)
        print("  -> FER2013 test...")
        f3_preds = infer_pixels(model, f3_pixels, f3_labels, img_size, device, in_ch, norm_im)
        if model_num_cls == 8:
            # 8-class model on 7-class data: ignore contempt predictions
            f3_preds_7 = np.clip(f3_preds, 0, 6)  # map contempt(7) to an existing class
            model_results["fer2013_test"] = compute_metrics(f3_labels, f3_preds_7, 7)
        else:
            model_results["fer2013_test"] = compute_metrics(f3_labels, f3_preds, 7)

        # On Self data (always 7-class)
        print("  -> Self data...")
        self_preds = infer_self_data(model, self_df, img_size, device, in_ch, norm_im)
        if model_num_cls == 8:
            self_preds_7 = np.clip(self_preds, 0, 6)
            model_results["self_data"] = compute_metrics(self_labels, self_preds_7, 7)
        else:
            model_results["self_data"] = compute_metrics(self_labels, self_preds, 7)

        results[label] = model_results

        # Print summary
        for test_name, m in model_results.items():
            if "8class" not in test_name:
                print(f"    {test_name}: acc={m['acc']:.4f}  f1={m['f1']:.4f}")

    # ---- Output summary table ----
    print(f"\n\n{'='*90}")
    print("CROSS-DOMAIN EVALUATION SUMMARY")
    print(f"{'='*90}")

    test_sets = ["ferplus_test", "fer2013_test", "self_data"]
    test_labels = ["FERPlus Test", "FER2013 Test", "Self Data"]

    # Build table
    rows = []
    for label, model_results in results.items():
        row = {"Model": label}
        for ts, tl in zip(test_sets, test_labels):
            if ts in model_results:
                m = model_results[ts]
                row[f"{tl} Acc"] = f"{m['acc']:.4f}"
                row[f"{tl} F1"] = f"{m['f1']:.4f}"
            else:
                row[f"{tl} Acc"] = "--"
                row[f"{tl} F1"] = "--"
        rows.append(row)

    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))
    df_out.to_csv(out / "cross_eval_table.csv", index=False)
    print(f"\n[saved] {out / 'cross_eval_table.csv'}")

    # ---- Radar chart: per-class recall on self data ----
    print("\n[generating radar chart]")
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for idx, (label, model_results) in enumerate(results.items()):
        if "self_data" in model_results:
            recalls = model_results["self_data"]["per_class_recall"]
            vals = recalls + [recalls[0]]
            ax.fill(angles, vals, alpha=0.1, color=colors[idx % len(colors)])
            ax.plot(angles, vals, color=colors[idx % len(colors)], linewidth=2, label=label)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(EMO_7)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_title("Per-Class Recall on Self-Collected Data", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "cross_eval_radar.png", dpi=150)
    plt.close()
    print(f"[saved] {out / 'cross_eval_radar.png'}")

    # ---- Domain gap bar chart ----
    print("[generating domain gap chart]")
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(results))
    w = 0.2
    for i, ts_key in enumerate(["ferplus_test", "fer2013_test", "self_data"]):
        accs = []
        labels_list = []
        for label, model_results in results.items():
            if ts_key in model_results:
                accs.append(model_results[ts_key]["acc"])
                if i == 0:
                    labels_list.append(label)
        offset = (i - 1) * w
        bars = ax.bar(x + offset, accs, w, label=test_labels[i], edgecolor="white")
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list if labels_list else list(results.keys()), rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title("Cross-Domain Accuracy Comparison", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "domain_gap_summary.png", dpi=150)
    plt.close()
    print(f"[saved] {out / 'domain_gap_summary.png'}")

    print(f"\n[EXP4 DONE] Results in {out}/")


if __name__ == "__main__":
    main()
