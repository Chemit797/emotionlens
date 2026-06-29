"""用同学模型 (ResNet18, 7-class, FERPlus+RAFDB+self) 在我们的 self 数据上评估。
同时评估 base (best.pt) 和 personalized (personalized.pt) 两个版本。"""
import torch
import torch.nn as nn
import torchvision.models as tvm
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EMO_7 = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
EMO_8 = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
CONTEMPT_IDX = 7
COLORS = plt.cm.tab10(np.linspace(0, 1, 8))

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 13, "axes.labelsize": 11,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.1,
})


def resolve_image_path(filepath):
    fp = filepath.replace("\\", "/")
    if fp.startswith("raw/"):
        fp_stripped = fp[len("raw/"):]
    else:
        fp_stripped = fp
    p = Path("self") / fp_stripped
    if p.exists():
        return p
    p2 = Path("user2_data") / fp
    if p2.exists():
        return p2
    return p


def build_resnet18(num_classes=7):
    m = tvm.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def load_classmate_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = build_resnet18(7).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    img_size = ckpt.get("img_size", 224)
    return model, img_size


def load_all_data():
    """加载所有 self 数据（含 user2_data 目录下的 user2）"""
    all_rows = []
    # self/ 下各用户
    for user_dir in sorted(Path("self").iterdir()):
        if not user_dir.is_dir() or user_dir.name.startswith("."):
            continue
        pl = user_dir / "prelabels.csv"
        if not pl.exists():
            continue
        df = pd.read_csv(pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        emo_map = {e: i for i, e in enumerate(EMO_8)}
        df["label"] = df["final_label"].map(emo_map)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        df["user"] = user_dir.name
        all_rows.append(df)

    # user2_data/ 独立目录
    u2_pl = Path("user2_data") / "prelabels.csv"
    if u2_pl.exists():
        df = pd.read_csv(u2_pl)
        if "reviewed" in df.columns:
            df = df[df["reviewed"] == True]
        emo_map = {e: i for i, e in enumerate(EMO_8)}
        df["label"] = df["final_label"].map(emo_map)
        df = df[df["label"].notna()].copy()
        df["label"] = df["label"].astype(int)
        df["user"] = "user2"
        all_rows.append(df)

    full_df = pd.concat(all_rows, ignore_index=True)
    return full_df


@torch.no_grad()
def evaluate_model(model, img_size, device, full_df):
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_preds = []
    for start in range(0, len(full_df), 256):
        batch = full_df.iloc[start:start + 256]
        batch_tensors = []
        for _, row in batch.iterrows():
            impath = resolve_image_path(row["filepath"])
            if impath.exists():
                try:
                    img = Image.open(impath).convert("RGB")
                    batch_tensors.append(preprocess(img))
                except Exception:
                    batch_tensors.append(torch.zeros(3, img_size, img_size))
            else:
                batch_tensors.append(torch.zeros(3, img_size, img_size))
        xb = torch.stack(batch_tensors).to(device)
        out = model(xb)
        all_preds.append(out.argmax(1).cpu().numpy())

    preds = np.concatenate(all_preds)  # 0-6 (7 classes)
    labels = full_df["label"].values  # 0-7 (8 classes)

    # 计算 correct: contempt 永远错，非 contempt 比较 pred==label
    correct = np.zeros(len(full_df), dtype=int)
    non_cont = labels != CONTEMPT_IDX
    correct[non_cont] = (preds[non_cont] == labels[non_cont]).astype(int)
    # contempt: correct stays 0

    # 构建 per-user per-emotion 数据
    users = sorted(full_df["user"].unique())
    table_data = {}
    for user in users:
        mask = (full_df["user"] == user).values
        u_labels = labels[mask]
        u_correct = correct[mask]
        accs = []
        for e in range(8):
            em = u_labels == e
            accs.append(float(u_correct[em].mean()) if em.sum() > 0 else float("nan"))
        table_data[user] = {
            "per_emotion": accs,
            "overall": float(u_correct.mean()),
            "n": int(mask.sum()),
        }

    overall_accs = []
    for e in range(8):
        em = labels == e
        overall_accs.append(float(correct[em].mean()) if em.sum() > 0 else float("nan"))
    table_data["Overall"] = {
        "per_emotion": overall_accs,
        "overall": float(correct.mean()),
        "n": len(full_df),
    }

    return table_data, users, overall_accs, correct


def print_table(table_data, users, title):
    print(f"\n{'='*110}")
    print(title)
    print(f"{'='*110}")
    hdr = f"{'User':>8s} | " + " | ".join(f"{e:>8s}" for e in EMO_8) + f" | {'Overall':>8s} | {'#Imgs':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for user in users:
        td = table_data[user]
        row = f"{user:>8s} | " + " | ".join(
            f"{a:8.3f}" if not np.isnan(a) else f"{'--':>8s}" for a in td["per_emotion"]
        ) + f" | {td['overall']:8.3f} | {td['n']:>6d}"
        print(row)
    print("-" * len(hdr))
    td = table_data["Overall"]
    row = f"{'Overall':>8s} | " + " | ".join(
        f"{a:8.3f}" if not np.isnan(a) else f"{'--':>8s}" for a in td["per_emotion"]
    ) + f" | {td['overall']:8.3f} | {td['n']:>6d}"
    print(row)


def save_charts(table_data, users, overall_accs, suffix, title_prefix):
    out_dir = Path("runs/self_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_data = np.array([
        table_data[u]["per_emotion"] for u in users
    ] + [overall_accs])
    row_labels = [f"{u} ({table_data[u]['n']})" for u in users] + \
                 [f"Overall ({table_data['Overall']['n']})"]

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(heatmap_data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    for i in range(len(row_labels)):
        for j in range(8):
            val = heatmap_data[i, j]
            txt = f"{val:.2f}" if not np.isnan(val) else "--"
            clr = "white" if not np.isnan(val) and val < 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color=clr)
    ax.set_xticks(range(8)); ax.set_xticklabels(EMO_8, rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    ax.set_title(f"{title_prefix} — Accuracy Heatmap", fontweight="bold", fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.85).set_label("Accuracy")
    plt.tight_layout()
    hp = out_dir / f"classmate_{suffix}_heatmap.png"
    plt.savefig(hp); plt.close()
    print(f"[saved] {hp}")

    # Grouped bars
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(users) + 1)
    w = 0.1
    for e_idx, emo in enumerate(EMO_8):
        vals = [table_data[u]["per_emotion"][e_idx] for u in users] + [overall_accs[e_idx]]
        ax.bar(x + (e_idx - 3.5) * w, vals, w, label=emo, color=COLORS[e_idx],
               edgecolor="white", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(row_labels, rotation=20, ha="right")
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.15)
    ax.set_title(f"{title_prefix} — Per User x Per Emotion", fontweight="bold")
    ax.legend(fontsize=7, ncol=4, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    bp = out_dir / f"classmate_{suffix}_bars.png"
    plt.savefig(bp); plt.close()
    print(f"[saved] {bp}")


def save_csv(table_data, users, suffix):
    rows = []
    for user in users:
        td = table_data[user]
        for e_idx, emo in enumerate(EMO_8):
            rows.append({
                "user": user, "emotion": emo,
                "accuracy": td["per_emotion"][e_idx],
                "overall": td["overall"], "n": td["n"],
            })
    pd.DataFrame(rows).to_csv(f"runs/self_eval/classmate_{suffix}.csv", index=False)
    print(f"[saved] runs/self_eval/classmate_{suffix}.csv")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # 加载数据
    full_df = load_all_data()
    print(f"[data] {len(full_df)} images, {full_df['user'].nunique()} users")
    for u in sorted(full_df["user"].unique()):
        print(f"  {u}: {len(full_df[full_df['user']==u])} images")

    EMOTIONZ = Path("D:/Project/xmum-2604/emotionz/emotionz/final_outputs")

    model_configs = [
        (EMOTIONZ / "best.pt", "base", "Classmate Base Model"),
        (EMOTIONZ / "personalized.pt", "personalized", "Classmate Personalized Model"),
    ]

    results = {}
    for ckpt_path, suffix, label in model_configs:
        if not ckpt_path.exists():
            print(f"[skip] {ckpt_path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {label} ({ckpt_path.name})")
        print(f"{'='*60}")
        model, img_size = load_classmate_model(ckpt_path, device)
        print(f"[model] ResNet18-7class, img_size={img_size}")
        table_data, users, overall_accs, correct = evaluate_model(model, img_size, device, full_df)
        print_table(table_data, users, label)
        save_charts(table_data, users, overall_accs, suffix, label)
        save_csv(table_data, users, suffix)
        results[suffix] = table_data

    # ====== 对比总结 ======
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON: Our Model vs Classmate (Base) vs Classmate (Personalized)")
    print(f"{'='*70}")

    # 加载我们的 per-user 数据
    our_csv = Path("runs/self_eval/per_user_emotion.csv")
    our_per_user = {}
    if our_csv.exists():
        odf = pd.read_csv(our_csv)
        for u in odf["user"].unique():
            udf = odf[odf["user"] == u]
            tot_correct = udf["correct"].sum()
            tot_total = udf["total"].sum()
            our_per_user[u] = {"overall": tot_correct / tot_total if tot_total > 0 else 0, "n": int(tot_total)}

    for suffix, label in [("base", "Base"), ("personalized", "Personalized")]:
        if suffix not in results:
            continue
        print(f"\n--- vs Classmate {label} ---")
        td = results[suffix]
        # Header
        print(f"{'User':>8s} | {'Ours':>8s} | {'Theirs':>8s} | {'Diff':>8s} | {'Winner':>10s} | {'#Imgs':>6s}")
        print("-" * 62)
        for user in sorted(td.keys()):
            if user == "Overall":
                continue
            our = our_per_user.get(user, {}).get("overall", float("nan"))
            their = td[user]["overall"]
            diff = their - our
            winner = "Classmate" if diff > 0 else ("Ours" if diff < 0 else "Tie")
            n = td[user]["n"]
            print(f"{user:>8s} | {our:8.3f} | {their:8.3f} | {diff:+8.3f} | {winner:>10s} | {n:>6d}")
        # Overall
        our_overall = sum(v["overall"] * v["n"] for v in our_per_user.values()) / \
                      sum(v["n"] for v in our_per_user.values()) if our_per_user else 0
        their_overall = td["Overall"]["overall"]
        print("-" * 62)
        print(f"{'Overall':>8s} | {our_overall:8.3f} | {their_overall:8.3f} | {their_overall-our_overall:+8.3f} | {'Classmate' if their_overall > our_overall else 'Ours':>10s} | {td['Overall']['n']:>6d}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
