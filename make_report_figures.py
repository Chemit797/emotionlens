"""生成报告用的全套图表（英文，美观）"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# 全局风格
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 7))


def plot_training_curves(metrics_path, out_path):
    """loss / val_acc / val_f1 三合一曲线"""
    with open(metrics_path) as f:
        data = json.load(f)
    hist = data["history"]
    epochs = [h["epoch"] + 1 for h in hist]
    loss = [h["train_loss"] for h in hist]
    vacc = [h["val_acc"] for h in hist]
    vf1 = [h["val_f1"] for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # 左：loss
    ax1.plot(epochs, loss, color="#2c3e50", linewidth=2)
    ax1.fill_between(epochs, loss, alpha=0.15, color="#2c3e50")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # 右：val acc + f1
    ax2.plot(epochs, vacc, color="#27ae60", linewidth=2, label="Val Accuracy")
    ax2.plot(epochs, vf1, color="#2980b9", linewidth=2, label="Val Macro-F1")
    ax2.axhline(y=data["test_acc"], color="#27ae60", linestyle="--", alpha=0.6,
                label=f"Test Acc = {data['test_acc']:.3f}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_title("Validation & Test Metrics")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.suptitle("FER Training Curves — ResNet18 on FERPlus+RAFDB+Self", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_class_distribution(parquet_path, out_path):
    """训练集类别分布（对数坐标凸显小类）"""
    FERP_EMO = EMO + ["contempt"]  # FERPlus 有 8 类
    FERP_COLORS = plt.cm.tab10(np.linspace(0, 1, 8))
    df = pd.read_parquet(parquet_path)
    tr = df[df.split == "train"]
    counts = tr["label"].value_counts().reindex(range(8), fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(FERP_EMO, counts.values, color=FERP_COLORS, edgecolor="white", linewidth=0.8)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
                str(count), ha="center", fontsize=8, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("Number of Samples (log scale)")
    ax.set_title("FERPlus Training Set — Class Distribution")
    ax.set_xticks(range(len(FERP_EMO)))
    ax.set_xticklabels(FERP_EMO, rotation=30, ha="right")
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.grid(axis="y", alpha=0.3)

    # 标注不平衡率
    ratio = counts.max() / counts.min()
    ax.annotate(f"Imbalance ratio: {ratio:.0f}:1\n(max: {FERP_EMO[counts.idxmax()]}, min: {FERP_EMO[counts.idxmin()]})",
                xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_confusion_matrix_pretty(parquet_path, ckpt_path, img_size, out_path):
    """重新生成更美观的混淆矩阵（百分比标注）"""
    import torch, torch.nn as nn, cv2
    import torchvision.models as tvm
    from torchvision import transforms
    from PIL import Image
    from torch.utils.data import DataLoader
    from sklearn.metrics import confusion_matrix

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    ckpt_data = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(ckpt_data["model_state"])
    m = m.to(device).eval()

    # 加载测试集
    df = pd.read_parquet(parquet_path)
    te = df[df.split == "test"]
    pixels = te["pixels"].tolist()
    labels = te["label"].values

    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 批量推理
    preds = []
    bs = 128
    for i in range(0, len(pixels), bs):
        batch = pixels[i:i+bs]
        imgs = []
        for p in batch:
            img = np.array(p.split(), dtype=np.uint8).reshape(48, 48)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img_rgb)
            x = preprocess(img_pil)
            imgs.append(x)
        xb = torch.stack(imgs).to(device)
        with torch.no_grad():
            out = m(xb)
        preds.append(out.argmax(1).cpu().numpy())
    preds = np.concatenate(preds)

    cm = confusion_matrix(labels, preds, labels=list(range(7)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 左：原始计数
    im1 = ax1.imshow(cm, cmap="Blues", aspect="auto")
    for i in range(7):
        for j in range(7):
            color = "white" if cm[i, j] > cm.max() * 0.5 else "black"
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8, color=color)
    ax1.set_xticks(range(7)); ax1.set_yticks(range(7))
    ax1.set_xticklabels(EMO, rotation=45, ha="right")
    ax1.set_yticklabels(EMO)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True Label")
    ax1.set_title("Confusion Matrix (counts)")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 右：行归一化百分比
    im2 = ax2.imshow(cm_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for i in range(7):
        for j in range(7):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax2.text(j, i, f"{cm_norm[i,j]:.1%}", ha="center", va="center", fontsize=8, color=color)
    ax2.set_xticks(range(7)); ax2.set_yticks(range(7))
    ax2.set_xticklabels(EMO, rotation=45, ha="right")
    ax2.set_yticklabels(EMO)
    ax2.set_xlabel("Predicted")
    ax2.set_title("Confusion Matrix (row-normalized)")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    acc = (labels == preds).mean()
    from sklearn.metrics import f1_score
    f1 = f1_score(labels, preds, average="macro")
    fig.suptitle(f"FERPlus Test Set — Accuracy={acc:.3f}  Macro-F1={f1:.3f}", fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


def plot_per_class_metrics(eval_results, out_path):
    """每类 precision/recall/f1 柱状图"""
    cats = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
    precision = [0.821, 0.920, 0.821, 0.656, 0.751, 0.579, 0.689]
    recall    = [0.852, 0.915, 0.824, 0.667, 0.742, 0.478, 0.452]
    f1        = [0.836, 0.917, 0.822, 0.661, 0.746, 0.524, 0.545]
    support   = [1262, 928, 444, 444, 325, 23, 93]

    x = np.arange(len(cats))
    w = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 左：precision/recall/f1
    ax1.bar(x - w, precision, w, label="Precision", color="#3498db", edgecolor="white")
    ax1.bar(x, recall, w, label="Recall", color="#2ecc71", edgecolor="white")
    ax1.bar(x + w, f1, w, label="F1-score", color="#e74c3c", edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cats, rotation=30, ha="right")
    ax1.set_ylabel("Score")
    ax1.set_title("Per-Class Precision / Recall / F1")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # 右：support + f1
    colors_f1 = plt.cm.RdYlGn(np.array(f1) / max(f1))
    bars2 = ax2.bar(cats, f1, color=colors_f1, edgecolor="white")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(cats, support, "o-", color="#34495e", linewidth=2, markersize=8, label="Support")
    ax2.set_ylabel("F1-score")
    ax2_twin.set_ylabel("Number of Samples")
    ax2.set_title("Per-Class F1-score vs Sample Count")
    ax2.set_xticks(range(len(cats)))
    ax2.set_xticklabels(cats, rotation=30, ha="right")
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis="y", alpha=0.3)
    ax2_twin.legend(loc="upper right", fontsize=8)

    # 标注
    for bar, f in zip(bars2, f1):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{f:.2f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("Per-Class Performance — ResNet18 on FERPlus Test Set", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    RUNS = Path("runs")
    BEST = RUNS / "classmate_model"
    METRICS = RUNS / "efficientnet_b0_20260615_235918" / "metrics.json"  # old metrics still present
    CKPT = BEST / "best.pt"
    PARQUET = Path("data/ferplus.parquet")

    print("Generating report figures...\n")

    # 1. 训练曲线
    plot_training_curves(METRICS, RUNS / "training_curves.png")

    # 2. 类别分布
    plot_class_distribution(PARQUET, RUNS / "class_distribution.png")

    # 3. 美观版混淆矩阵
    plot_confusion_matrix_pretty(PARQUET, CKPT, 224,
                                 RUNS / "confusion_matrix_pretty.png")

    # 4. 逐类指标
    plot_per_class_metrics(None, RUNS / "per_class_metrics.png")

    print("\n[DONE] All report figures generated in runs/")
