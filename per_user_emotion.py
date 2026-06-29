"""按组员 × 表情 细分准确率
Model: ResNet18 (7-class), FERPlus + RAFDB + self"""
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
import cv2, numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
COLORS = plt.cm.tab10(np.linspace(0, 1, 7))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


def resolve_image_path(filepath, self_dir="self"):
    fp = filepath.replace("\\", "/")
    if fp.startswith("raw/"):
        fp_stripped = fp[len("raw/"):]
    else:
        fp_stripped = fp
    p = Path(self_dir) / fp_stripped
    if p.exists():
        return p
    # user2: 原始路径在 user2_data/ 下
    p2 = Path("user2_data") / fp
    if p2.exists():
        return p2
    return p  # 可能不存在，调用方用 zeros 替代


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # 加载模型
    ckpt = "runs/classmate_model/best.pt"
    ckpt_data = torch.load(ckpt, map_location=device)
    img_size = ckpt_data.get("img_size", 224)

    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 7)
    m.load_state_dict(ckpt_data["model_state"])
    m = m.to(device).eval()
    print(f"[model] ResNet18 7-class  img_size={img_size}")

    # 加载所有组员数据
    all_rows = []
    for user_dir in sorted(Path("self").iterdir()):
        if not user_dir.is_dir() or user_dir.name.startswith("."):
            continue
        pl = user_dir / "prelabels.csv"
        if not pl.exists():
            continue
        df = pd.read_csv(pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        # 字符串标签 → 整数
        emo_map = {e: i for i, e in enumerate(EMO)}
        df["label"] = df["final_label"].map(emo_map)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        df["user"] = user_dir.name
        all_rows.append(df)

    # user2: 独立目录
    u2_pl = Path("user2_data") / "prelabels.csv"
    if u2_pl.exists():
        df = pd.read_csv(u2_pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        emo_map = {e: i for i, e in enumerate(EMO)}
        df["label"] = df["final_label"].map(emo_map)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        df["user"] = "user2"
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)
    print(f"[data] {len(full_df)} images, {full_df['user'].nunique()} users")

    # 预处理（RGB ImageNet 归一化）
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 逐张推理
    all_preds = []
    for start in range(0, len(full_df), 256):
        batch = full_df.iloc[start:start + 256]
        batch_imgs = []
        for _, row in batch.iterrows():
            impath = resolve_image_path(row["filepath"])
            if impath.exists():
                img = cv2.imread(str(impath))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_pil = Image.fromarray(img_rgb)
                    x = preprocess(img_pil)
                    batch_imgs.append(x)
                else:
                    batch_imgs.append(torch.zeros(3, img_size, img_size))
            else:
                batch_imgs.append(torch.zeros(3, img_size, img_size))
        xb = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            out = m(xb)
        all_preds.append(out.argmax(1).cpu().numpy())

    full_df["pred"] = np.concatenate(all_preds)
    full_df["correct"] = (full_df["label"] == full_df["pred"]).astype(int)

    users = sorted(full_df["user"].unique())

    # ========= 表格：每个组员 × 每类表情的准确率 =========
    print(f"\n{'='*110}")
    print(f"Per-User × Per-Emotion Accuracy Table")
    print(f"{'='*110}")
    header = f"{'User':>8s} | " + " | ".join(f"{e:>8s}" for e in EMO) + f" | {'Overall':>8s} | {'#Imgs':>6s}"
    print(header)
    print("-" * len(header))

    table_data = {}  # for plotting
    for user in users:
        u_mask = full_df["user"] == user
        u_df = full_df[u_mask]
        accs = []
        for e_idx in range(7):
            e_mask = u_df["label"] == e_idx
            if e_mask.sum() > 0:
                accs.append(u_df[e_mask]["correct"].mean())
            else:
                accs.append(float("nan"))
        overall = u_df["correct"].mean()
        n = len(u_df)
        row = f"{user:>8s} | " + " | ".join(
            f"{a:8.3f}" if not np.isnan(a) else f"{'--':>8s}" for a in accs
        ) + f" | {overall:8.3f} | {n:>6d}"
        print(row)
        table_data[user] = {"per_emotion": accs, "overall": overall, "n": n}

    # 整体
    print("-" * len(header))
    overall_accs = []
    for e_idx in range(7):
        e_mask = full_df["label"] == e_idx
        overall_accs.append(full_df[e_mask]["correct"].mean() if e_mask.sum() > 0 else float("nan"))
    overall_all = full_df["correct"].mean()
    row = f"{'Overall':>8s} | " + " | ".join(
        f"{a:8.3f}" if not np.isnan(a) else f"{'--':>8s}" for a in overall_accs
    ) + f" | {overall_all:8.3f} | {len(full_df):>6d}"
    print(row)
    table_data["Overall"] = {"per_emotion": overall_accs, "overall": overall_all, "n": len(full_df)}

    # ========= 图1：热力图 =========
    fig, ax = plt.subplots(figsize=(14, 5))
    heatmap_data = []
    row_labels = []
    for user in users:
        heatmap_data.append(table_data[user]["per_emotion"])
        row_labels.append(f"{user} ({table_data[user]['n']})")
    heatmap_data.append(overall_accs)
    row_labels.append(f"Overall ({len(full_df)})")

    heatmap_data = np.array(heatmap_data)
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(len(row_labels)):
        for j in range(7):
            val = heatmap_data[i, j]
            text = f"{val:.2f}" if not np.isnan(val) else "--"
            color = "white" if not np.isnan(val) and val < 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, fontweight="bold", color=color)
    ax.set_xticks(range(7)); ax.set_xticklabels(EMO, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title("Per-User × Per-Emotion Accuracy Heatmap", fontweight="bold", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.85).set_label("Accuracy")
    plt.tight_layout()
    plt.savefig("runs/self_eval/per_user_emotion_heatmap.png")
    plt.close()
    print(f"\n[saved] runs/self_eval/per_user_emotion_heatmap.png")

    # ========= 图2：分组柱状图 =========
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(users) + 1)
    w = 0.1
    for e_idx, emo in enumerate(EMO):
        vals = [table_data[u]["per_emotion"][e_idx] for u in users] + [overall_accs[e_idx]]
        offset = (e_idx - 3) * w
        bars = ax.bar(x + offset, vals, w, label=emo, color=COLORS[e_idx], edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(row_labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title("Per-User × Per-Emotion Accuracy — Grouped Bars", fontweight="bold")
    ax.legend(fontsize=7, ncol=4, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("runs/self_eval/per_user_emotion_bars.png")
    plt.close()
    print(f"[saved] runs/self_eval/per_user_emotion_bars.png")

    # 保存CSV
    csv_data = {"user": [], "emotion": [], "accuracy": [], "correct": [], "total": []}
    for user in users:
        u_mask = full_df["user"] == user
        u_df = full_df[u_mask]
        for e_idx, emo in enumerate(EMO):
            e_mask = u_df["label"] == e_idx
            csv_data["user"].append(user)
            csv_data["emotion"].append(emo)
            csv_data["accuracy"].append(u_df[e_mask]["correct"].mean() if e_mask.sum() > 0 else float("nan"))
            csv_data["correct"].append(int(u_df[e_mask]["correct"].sum()) if e_mask.sum() > 0 else 0)
            csv_data["total"].append(int(e_mask.sum()))
    pd.DataFrame(csv_data).to_csv("runs/self_eval/per_user_emotion.csv", index=False)
    print(f"[saved] runs/self_eval/per_user_emotion.csv")


if __name__ == "__main__":
    main()
