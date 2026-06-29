"""对自采组员数据做全面评估 + 图表 + prototypes
Model: ResNet18 (7-class), FERPlus + RAFDB + self"""
import json, time, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def resolve_image_path(filepath, self_dir="self"):
    """将 prelabels.csv 中的 filepath 映射到实际图片路径。"""
    fp = filepath.replace("\\", "/")
    if fp.startswith("raw/"):
        fp_stripped = fp[len("raw/"):]
    else:
        fp_stripped = fp
    p = Path(self_dir) / fp_stripped
    if p.exists():
        return p
    # fallback: user2_data 独立目录
    p2 = Path("user2_data") / fp
    if p2.exists():
        return p2
    return p
import pandas as pd
import torch, torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import f1_score, confusion_matrix, classification_report

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
EMO_CN = {"neutral": "平静", "happiness": "开心", "surprise": "惊讶", "sadness": "悲伤",
          "anger": "愤怒", "disgust": "厌恶", "fear": "恐惧"}
COLORS = plt.cm.tab10(np.linspace(0, 1, 7))
EMO_COLORS_STRONG = ["#7f8c8d", "#27ae60", "#f39c12", "#2980b9", "#e74c3c", "#2ecc71", "#8e44ad"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


def load_model(ckpt, device):
    """Load ResNet18 7-class model."""
    ckpt_data = torch.load(ckpt, map_location=device)
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(ckpt_data["model_state"])
    return m.to(device).eval()


def load_all_self_data(self_dir):
    """加载所有组员的 reviewed 数据"""
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

    # 也加载 user2_data（独立目录）
    u2_pl = Path("user2_data/prelabels.csv")
    if u2_pl.exists():
        df = pd.read_csv(u2_pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        df["user"] = "user2"
        rows.append(df)

    all_df = pd.concat(rows, ignore_index=True)
    print(f"[data] {len(all_df)} reviewed images from {all_df['user'].nunique()} users")
    return all_df


def preprocess_pixels(pixels, img_size):
    """像素字符串 → tensor (RGB ImageNet 归一化)"""
    img = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_pil = Image.fromarray(img_rgb)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img_pil)


@torch.no_grad()
def run_inference(model, df, img_size, device):
    """对全部自采数据做推理"""
    all_probs = []
    bs = 256
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    pixels = df["pixels"].tolist() if "pixels" in df.columns else None

    if pixels is None:
        # 从 filepath 加载图片
        for start in range(0, len(df), bs):
            batch = df.iloc[start:start+bs]
            batch_imgs = []
            for _, row in batch.iterrows():
                impath = resolve_image_path(row["filepath"])
                img = cv2.imread(str(impath))
                if img is None:
                    batch_imgs.append(torch.zeros(3, img_size, img_size))
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                x = preprocess(img_pil)
                batch_imgs.append(x)
            xb = torch.stack(batch_imgs).to(device)
            out = model(xb)
            probs = torch.softmax(out, 1).cpu().numpy()
            all_probs.append(probs)
    else:
        for start in range(0, len(df), bs):
            batch = pixels[start:start+bs]
            batch_imgs = [preprocess_pixels(p, img_size) for p in batch]
            xb = torch.stack(batch_imgs).to(device)
            out = model(xb)
            probs = torch.softmax(out, 1).cpu().numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs)
    all_preds = all_probs.argmax(1)
    return all_probs, all_preds


def plot_self_confusion_matrix(labels, preds, out_path):
    """自采数据混淆矩阵"""
    cm = confusion_matrix(labels, preds, labels=list(range(7)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(7):
        for j in range(7):
            c = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color=c)
    ax1.set_xticks(range(7)); ax1.set_yticks(range(7))
    ax1.set_xticklabels(EMO, rotation=45, ha="right"); ax1.set_yticklabels(EMO)
    ax1.set_xlabel("Predicted"); ax1.set_ylabel("True Label")
    ax1.set_title("Self-Data Confusion Matrix (counts)")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    im2 = ax2.imshow(cm_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            c = "white" if cm_norm[i, j] > 0.55 else "black"
            ax2.text(j, i, f"{cm_norm[i,j]:.1%}", ha="center", va="center", fontsize=8, color=c)
    ax2.set_xticks(range(7)); ax2.set_yticks(range(7))
    ax2.set_xticklabels(EMO, rotation=45, ha="right"); ax2.set_yticklabels(EMO)
    ax2.set_xlabel("Predicted")
    ax2.set_title("Self-Data Confusion Matrix (row-normalized)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    acc = (labels == preds).mean()
    f1 = f1_score(labels, preds, average="macro")
    fig.suptitle(f"Team Self-Collected Data — Acc={acc:.3f}  Macro-F1={f1:.3f}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_per_user_metrics(df, probs, preds, out_path):
    """按组员拆分的性能"""
    users = sorted(df["user"].unique())
    user_acc = []
    user_f1 = []
    for u in users:
        mask = df["user"] == u
        u_labels = df.loc[mask, "label"].values
        u_preds = preds[mask]
        acc = (u_labels == u_preds).mean()
        f1 = f1_score(u_labels, u_preds, average="macro", zero_division=0)
        user_acc.append(acc)
        user_f1.append(f1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(users))
    w = 0.35
    bars1 = ax1.bar(x - w/2, user_acc, w, label="Accuracy", color="#3498db", edgecolor="white")
    bars2 = ax1.bar(x + w/2, user_f1, w, label="Macro-F1", color="#e74c3c", edgecolor="white")
    for bar, val in zip(bars1, user_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    for bar, val in zip(bars2, user_f1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_xticks(x); ax1.set_xticklabels(users)
    ax1.set_ylabel("Score"); ax1.set_ylim(0, 1.15)
    ax1.set_title("Per-User Accuracy & Macro-F1")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # 右：每个用户样本数 + 各表情分布堆叠
    for i, u in enumerate(users):
        bottom = 0
        for e_idx, emo in enumerate(EMO):
            cnt = ((df["user"] == u) & (df["label"] == e_idx)).sum()
            if cnt > 0:
                ax2.bar(i, cnt, bottom=bottom, color=COLORS[e_idx], edgecolor="white", linewidth=0.3)
                bottom += cnt
    ax2.set_xticks(x); ax2.set_xticklabels(users)
    ax2.set_ylabel("Number of Images")
    ax2.set_title("Per-User Sample Distribution")
    ax2.legend(EMO, fontsize=7, loc="upper right", ncol=2)

    fig.suptitle("Per-User Performance — Self-Collected Data", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")

    return users, user_acc, user_f1


def plot_domain_gap(ferplus_metrics_path, self_acc, self_f1, out_path):
    """FERPlus vs 自采数据 domain gap 对比"""
    with open(ferplus_metrics_path) as f:
        fp = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # 左：acc 对比
    bars = ax1.bar(["FERPlus Test", "Self-Collected"],
                   [fp["test_acc"], self_acc],
                   color=["#3498db", "#e74c3c"], edgecolor="white", width=0.4)
    for bar, val in zip(bars, [fp["test_acc"], self_acc]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Accuracy"); ax1.set_ylim(0, 1.1)
    ax1.set_title("Accuracy: FERPlus vs Self-Collected")
    ax1.grid(axis="y", alpha=0.3)

    # 右：F1 对比
    bars2 = ax2.bar(["FERPlus Test", "Self-Collected"],
                    [fp["test_f1"], self_f1],
                    color=["#3498db", "#e74c3c"], edgecolor="white", width=0.4)
    for bar, val in zip(bars2, [fp["test_f1"], self_f1]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Macro-F1"); ax2.set_ylim(0, 1.1)
    ax2.set_title("Macro-F1: FERPlus vs Self-Collected")
    ax2.grid(axis="y", alpha=0.3)

    gap_acc = fp["test_acc"] - self_acc
    gap_f1 = fp["test_f1"] - self_f1
    fig.suptitle(f"Domain Gap  ΔAcc={gap_acc:+.3f}  ΔF1={gap_f1:+.3f}",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_per_emotion_comparison(ferplus_cm_norm, self_cm_norm, out_path):
    """FERPlus vs Self 每类 recall 对比"""
    fp_recall = np.diag(ferplus_cm_norm)
    self_recall = np.diag(self_cm_norm)

    x = np.arange(len(EMO))
    w = 0.3
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, fp_recall, w, label="FERPlus Test", color="#3498db", edgecolor="white")
    ax.bar(x + w/2, self_recall, w, label="Self-Collected", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(EMO, rotation=30, ha="right")
    ax.set_ylabel("Recall (per class)"); ax.set_ylim(0, 1.15)
    ax.set_title("Per-Class Recall: FERPlus vs Self-Collected")
    ax.legend(); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_ferplus_self_radar(ferplus_cm_norm, self_cm_norm, out_path):
    """FERPlus vs Self 雷达图对比"""
    fp_recall = np.diag(ferplus_cm_norm)
    self_recall = np.diag(self_cm_norm)

    angles = np.linspace(0, 2 * np.pi, len(EMO), endpoint=False).tolist()
    angles += angles[:1]

    fp_vals = fp_recall.tolist() + [fp_recall[0]]
    self_vals = self_recall.tolist() + [self_recall[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.fill(angles, fp_vals, alpha=0.25, color="#3498db", label="FERPlus Test")
    ax.plot(angles, fp_vals, color="#3498db", linewidth=2)
    ax.fill(angles, self_vals, alpha=0.25, color="#e74c3c", label="Self-Collected")
    ax.plot(angles, self_vals, color="#e74c3c", linewidth=2)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(EMO, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"])
    ax.set_title("Per-Class Recall — FERPlus vs Self-Collected", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def make_self_prototypes(df, probs, preds, out_path, top_k=5):
    """在自采数据上找每类最典型样本"""
    fig, axes = plt.subplots(7, top_k, figsize=(top_k * 2, 14))
    for cls_idx in range(7):
        mask = preds == cls_idx
        cls_probs = probs[mask, cls_idx]
        if len(cls_probs) == 0:
            for k in range(top_k):
                axes[cls_idx, k].axis("off")
            continue
        idx_in_mask = np.argsort(cls_probs)[::-1][:top_k]
        orig_indices = np.where(mask)[0][idx_in_mask]

        for k, orig_idx in enumerate(orig_indices):
            ax = axes[cls_idx, k]
            row = df.iloc[orig_idx]
            impath = resolve_image_path(row["filepath"])
            if impath.exists():
                img = cv2.imread(str(impath))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
            true_label = row["label"]
            true_emo = EMO[true_label]
            conf = probs[orig_idx, cls_idx]
            color = "green" if true_label == cls_idx else "red"
            ax.set_title(f"conf:{conf:.2f} | true:{true_emo}", fontsize=7,
                         color=color, fontweight="bold" if true_label == cls_idx else "normal")
            ax.axis("off")
            if k == 0:
                ax.set_ylabel(EMO[cls_idx], fontsize=12, fontweight="bold",
                              rotation=0, labelpad=40, va="center")

    plt.suptitle(f"Top-{top_k} Prototypical Examples — Self-Collected Data (green=correct, red=mislabeled)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def make_misclassification_grid(df, probs, preds, out_path, n_per_class=3):
    """每类找 model 最 '纠结' 的错误案例"""
    fig, axes = plt.subplots(7, n_per_class, figsize=(n_per_class * 2.5, 14))
    for cls_idx in range(7):
        # 找真实标签是该类但预测错了的
        mask = (df["label"].values == cls_idx) & (preds != cls_idx)
        if mask.sum() == 0:
            for k in range(n_per_class):
                axes[cls_idx, k].axis("off")
            continue
        # 按该类的 softmax 置信度升序（模型最不确定这是该类）
        cls_probs = probs[mask, cls_idx]
        idx_sorted = np.argsort(cls_probs)[:n_per_class]
        orig_indices = np.where(mask)[0][idx_sorted]

        for k, orig_idx in enumerate(orig_indices):
            ax = axes[cls_idx, k]
            row = df.iloc[orig_idx]
            impath = resolve_image_path(row["filepath"])
            if impath.exists():
                img = cv2.imread(str(impath))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img)
            true_emo = EMO[row["label"]]
            pred_emo = EMO[preds[orig_idx]]
            ax.set_title(f"pred:{pred_emo} | true:{true_emo}", fontsize=7, color="red")
            ax.axis("off")
            if k == 0:
                ax.set_ylabel(EMO[cls_idx], fontsize=11, fontweight="bold",
                              rotation=0, labelpad=40, va="center")

    plt.suptitle("Hardest Misclassifications — Self-Collected Data (pred vs true)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/classmate_model/best.pt")
    ap.add_argument("--self-dir", default="self")
    ap.add_argument("--ferplus-metrics", default="runs/efficientnet_b0_20260615_235918/metrics.json")
    ap.add_argument("--ferplus-parquet", default="data/ferplus.parquet")
    ap.add_argument("--ferplus-img-size", type=int, default=224)
    ap.add_argument("--out-dir", default="runs/self_eval")
    ap.add_argument("--top-k", type=int, default=5)
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    model = load_model(a.ckpt, device)
    img_size = 224
    print(f"[model] ResNet18 7-class  img_size={img_size}")

    # 加载数据
    df = load_all_self_data(a.self_dir)

    # 检查数据格式
    if "label" in df.columns and "final_label" in df.columns:
        # prelabels.csv format: final_label is the string name
        pass
    if "final_label" in df.columns:
        # 把字符串标签转为整数
        emo_map = {e: i for i, e in enumerate(EMO)}
        df["label"] = df["final_label"].map(emo_map)
        # 剔除无法映射的
        df = df[df["label"].notna()].reset_index(drop=True)
        df["label"] = df["label"].astype(int)

    print(f"[data] labels: {df['label'].value_counts().sort_index().to_dict()}")

    # 推理
    print("[inference] running...")
    probs, preds = run_inference(model, df, img_size, device)

    labels = df["label"].values
    acc = (labels == preds).mean()
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds, labels=list(range(7)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    print(f"\n{'='*50}")
    print(f"  Overall Accuracy: {acc:.4f}")
    print(f"  Macro F1:         {f1:.4f}")
    print(f"{'='*50}\n")

    print(classification_report(labels, preds, target_names=EMO, zero_division=0))

    # ========= 按用户 × 表情细分 =========
    df_eval = df.copy()
    df_eval["pred"] = preds
    df_eval["correct"] = (df_eval["label"] == df_eval["pred"]).astype(int)

    # 保存逐张预测结果
    df_eval[["user", "filepath", "final_label", "label", "pred", "correct"]].to_csv(
        out / "per_image_predictions.csv", index=False)
    print(f"[saved] {out / 'per_image_predictions.csv'}")

    # 打印表格
    users_sorted = sorted(df_eval["user"].unique())
    print(f"\n{'='*110}")
    print("Per-User × Per-Emotion Accuracy Table")
    print(f"{'='*110}")
    header = f"{'User':>8s} | " + " | ".join(f"{e:>8s}" for e in EMO) + f" | {'Overall':>8s} | {'#Imgs':>6s}"
    print(header)
    print("-" * len(header))

    table_data = {}
    for user in users_sorted:
        u_mask = df_eval["user"] == user
        u_df = df_eval[u_mask]
        accs = []
        for e_idx in range(7):
            e_mask = u_df["label"] == e_idx
            accs.append(u_df[e_mask]["correct"].mean() if e_mask.sum() > 0 else float("nan"))
        overall = u_df["correct"].mean()
        n = len(u_df)
        row = f"{user:>8s} | " + " | ".join(
            f"{a:8.3f}" if not np.isnan(a) else f"{'--':>8s}" for a in accs
        ) + f" | {overall:8.3f} | {n:>6d}"
        print(row)
        table_data[user] = {"per_emotion": accs, "overall": overall, "n": n}

    overall_accs = []
    for e_idx in range(7):
        e_mask = df_eval["label"] == e_idx
        overall_accs.append(df_eval[e_mask]["correct"].mean() if e_mask.sum() > 0 else float("nan"))
    overall_all = df_eval["correct"].mean()
    print("-" * len(header))
    row = f"{'Overall':>8s} | " + " | ".join(
        f"{a:8.3f}" if not np.isnan(a) else f"{'--':>8s}" for a in overall_accs
    ) + f" | {overall_all:8.3f} | {len(df_eval):>6d}"
    print(row)
    table_data["Overall"] = {"per_emotion": overall_accs, "overall": overall_all, "n": len(df_eval)}

    # 热力图
    fig, ax = plt.subplots(figsize=(14, 5))
    hm_data = np.array([table_data[u]["per_emotion"] for u in users_sorted] + [overall_accs])
    hm_labels = [f"{u} ({table_data[u]['n']})" for u in users_sorted] + [f"Overall ({len(df_eval)})"]
    im = ax.imshow(hm_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(len(hm_labels)):
        for j in range(7):
            val = hm_data[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "--"
            color = "white" if not np.isnan(val) and val < 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    ax.set_xticks(range(7)); ax.set_xticklabels(EMO, rotation=45, ha="right")
    ax.set_yticks(range(len(hm_labels))); ax.set_yticklabels(hm_labels)
    ax.set_title("Per-User × Per-Emotion Accuracy Heatmap", fontweight="bold", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.85).set_label("Accuracy")
    plt.tight_layout()
    plt.savefig(out / "per_user_emotion_heatmap.png")
    plt.close()
    print(f"[saved] {out / 'per_user_emotion_heatmap.png'}")

    # 分组柱状图
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(hm_labels))
    w = 0.1
    for e_idx, emo in enumerate(EMO):
        vals = [table_data[u]["per_emotion"][e_idx] for u in users_sorted] + [overall_accs[e_idx]]
        offset = (e_idx - 3) * w
        ax.bar(x + offset, vals, w, label=emo, color=plt.cm.tab10(e_idx/7), edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(hm_labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Per-User × Per-Emotion Accuracy — Grouped Bars", fontweight="bold")
    ax.legend(fontsize=7, ncol=4, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "per_user_emotion_bars.png")
    plt.close()
    print(f"[saved] {out / 'per_user_emotion_bars.png'}")
    # ========= 细分结束 =========

    # 保存数值结果
    per_class = {}
    for i, e in enumerate(EMO):
        mask_t = labels == i
        if mask_t.sum() > 0:
            per_class[e] = {
                "support": int(mask_t.sum()),
                "accuracy": float((labels[mask_t] == preds[mask_t]).mean()),
                "precision": float(((preds == i) & (labels == i)).sum() / max((preds == i).sum(), 1)),
                "recall": float(((preds == i) & (labels == i)).sum() / max((labels == i).sum(), 1)),
            }

    results = {
        "dataset": "self-collected",
        "n_images": len(df),
        "n_users": int(df["user"].nunique()),
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "per_class": per_class,
    }
    with open(out / "self_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ============ 图表 ============
    print("\n[generating plots]")

    # 1. 自采数据混淆矩阵
    plot_self_confusion_matrix(labels, preds, out / "self_confusion_matrix.png")

    # 2. 按组员指标
    users, uacc, uf1 = plot_per_user_metrics(df, probs, preds, out / "per_user_metrics.png")

    # 3. Domain Gap 对比
    plot_domain_gap(a.ferplus_metrics, acc, f1, out / "domain_gap.png")

    # 4. FERPlus vs Self 每类 recall 对比
    # 加载 FERPlus 测试集归一化混淆矩阵（重新算或从文件读）
    if Path(a.ferplus_parquet).exists():
        fp_df = pd.read_parquet(a.ferplus_parquet)
        fp_te = fp_df[fp_df.split == "test"]
        print(f"[ferplus] Computing FERPlus test confusion matrix on {len(fp_te)} images...")
        fp_probs, fp_preds = run_inference(model, fp_te, img_size, device)
        fp_cm = confusion_matrix(fp_te["label"].values, fp_preds, labels=list(range(7)))
        fp_cm_norm = fp_cm.astype(float) / fp_cm.sum(axis=1, keepdims=True).clip(min=1)
    else:
        # fallback: use diagonal from classification_report
        fp_cm_norm = np.diag([0.852, 0.915, 0.824, 0.667, 0.742, 0.478, 0.452])

    plot_per_emotion_comparison(fp_cm_norm, cm_norm, out / "per_emotion_recall_comparison.png")
    plot_ferplus_self_radar(fp_cm_norm, cm_norm, out / "recall_radar.png")

    # 5. 自采数据 Prototypes
    make_self_prototypes(df, probs, preds, out / "self_prototypes.png", a.top_k)

    # 6. 误分类案例
    make_misclassification_grid(df, probs, preds, out / "misclassifications.png", n_per_class=3)

    # ============ 文本摘要 ============
    with open(out / "self_eval_summary.md", "w") as f:
        f.write(f"# Self-Collected Data Evaluation Summary\n\n")
        f.write(f"- **Images**: {len(df)}\n")
        f.write(f"- **Users**: {df['user'].nunique()}\n")
        f.write(f"- **Accuracy**: {acc:.4f}\n")
        f.write(f"- **Macro-F1**: {f1:.4f}\n\n")
        f.write(f"## Per-User\n\n")
        f.write(f"| User | Images | Accuracy | Macro-F1 |\n")
        f.write(f"|------|--------|----------|----------|\n")
        for user, acc_u, f1_u in zip(users, uacc, uf1):
            n = (df['user'] == user).sum()
            f.write(f"| {user} | {n} | {acc_u:.4f} | {f1_u:.4f} |\n")
        f.write(f"\n## Per-Emotion\n\n")
        f.write(f"| Emotion | Support | Precision | Recall | F1 |\n")
        f.write(f"|---------|---------|-----------|--------|----|\n")
        for e, v in per_class.items():
            prec = v["precision"]
            rec = v["recall"]
            f1_val = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            f.write(f"| {e} | {v['support']} | {prec:.3f} | {rec:.3f} | {f1_val:.3f} |\n")

    print(f"\n[DONE] All results in {out}/")
    print(f"  - self_metrics.json")
    print(f"  - self_confusion_matrix.png")
    print(f"  - per_user_metrics.png")
    print(f"  - domain_gap.png")
    print(f"  - per_emotion_recall_comparison.png")
    print(f"  - recall_radar.png")
    print(f"  - self_prototypes.png")
    print(f"  - misclassifications.png")
    print(f"  - self_eval_summary.md")


if __name__ == "__main__":
    main()
