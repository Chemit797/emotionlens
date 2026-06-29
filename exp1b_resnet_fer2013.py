"""Experiment 1b: ResNet18 on FER2013 (7-class hard labels).

CONTROL VARIABLE: Architecture only.
  - Same data as Exp1 (FER2013 CSV, hard labels)
  - Same training recipe (lr=3e-4, epochs=35, AdamW, CosineAnnealing)
  - Same augmentation, same loss (hard CE)
  - ONLY change: VGG16 → ResNet18

This isolates architecture contribution:
  Exp1 (VGG16+FER2013)  →  11.6% self
  Exp1b (ResNet18+FER2013) →  ??.?% self   ← architecture delta
  Exp2 (ResNet18+FERPlus) →  44.4% self    ← label-quality delta (fer2013→ferplus)
  Exp3 (ResNet18+F+R+S)   →  64.3% self    ← data-diversity delta
"""
import json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2, random

EMO_7 = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]


# -------------------------------------------------------------------
# FER2013 Dataset (hard labels, 7-class) — identical to Exp1
# -------------------------------------------------------------------
class FER2013Dataset(Dataset):
    """Loads FER2013 CSV with columns: emotion, pixels, Usage.
    emotion: 0-6 (7 classes matching EMO_7)
    pixels: space-separated 48x48 grayscale values
    """
    def __init__(self, df, img_size=224, train=True):
        self.pixels = df["pixels"].tolist()
        self.labels = df["emotion"].to_numpy().astype(np.int64)
        self.img_size = img_size
        self.train = train

    def __len__(self):
        return len(self.pixels)

    def _img(self, i):
        return np.array(self.pixels[i].split(), dtype=np.uint8).reshape(48, 48)

    def _augment(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        if random.random() < 0.5:
            M = cv2.getRotationMatrix2D((24, 24), random.uniform(-12, 12), 1.0)
            img = cv2.warpAffine(img, M, (48, 48), borderMode=cv2.BORDER_REFLECT)
        if random.random() < 0.5:
            a, b = random.uniform(0.8, 1.2), random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, i):
        img = self._img(i)
        if self.train:
            img = self._augment(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = np.stack([img, img, img], axis=-1)
        x = torch.from_numpy(img).float().permute(2, 0, 1).div_(255.0)
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return x, int(self.labels[i])


# -------------------------------------------------------------------
# Data loaders — identical to Exp1
# -------------------------------------------------------------------
def make_loaders(csv_path, img_size, bs, nw):
    df = pd.read_csv(csv_path)
    df.columns = ["emotion", "pixels", "Usage"]
    tr = df[df.Usage == "Training"].reset_index(drop=True)
    va = df[df.Usage == "PublicTest"].reset_index(drop=True)
    te = df[df.Usage == "PrivateTest"].reset_index(drop=True)
    print(f"[data] train={len(tr)} val={len(va)} test={len(te)}")

    tr_ds = FER2013Dataset(tr, img_size, train=True)
    va_ds = FER2013Dataset(va, img_size, train=False)
    te_ds = FER2013Dataset(te, img_size, train=False)

    counts = np.bincount(tr["emotion"].to_numpy(), minlength=7)
    w = 1.0 / np.clip(counts.astype(np.float64), 1, None)
    sw = w[tr["emotion"].to_numpy()]
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True)

    return (DataLoader(tr_ds, bs, sampler=sampler, num_workers=nw, pin_memory=True, drop_last=True),
            DataLoader(va_ds, bs, shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(te_ds, bs, shuffle=False, num_workers=nw, pin_memory=True))


# -------------------------------------------------------------------
# Evaluation — identical to Exp1
# -------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    acc = (y == p).mean()
    f1 = f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p, labels=list(range(7)))
    return acc, f1, cm


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fer2013.csv")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}  {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}", flush=True)

    tl, vl, te = make_loaders(a.data, a.img_size, a.bs, a.workers)

    # ---- ONLY CHANGE from Exp1: VGG16 → ResNet18 ----
    model = tvm.resnet18(weights="IMAGENET1K_V1")
    model.fc = nn.Linear(512, 7)
    model = model.to(device)
    print(f"[model] ResNet18  params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    out = Path("runs") / f"resnet18_fer2013_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    best, hist = 0.0, []

    for ep in range(a.epochs):
        model.train()
        run, tot = 0.0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * x.size(0)
            tot += x.size(0)
        sched.step()
        vacc, vf1, _ = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_loss": run / tot, "val_acc": float(vacc), "val_f1": float(vf1)})
        print(f"ep{ep:02d} loss {run/tot:.3f}  val_acc {vacc:.4f}  val_f1 {vf1:.4f}")
        if vacc > best:
            best = vacc
            torch.save({"model_state": model.state_dict(), "img_size": a.img_size}, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    tacc, tf1, cm = evaluate(model, te, device)
    json.dump({"val_best": float(best), "test_acc": float(tacc), "test_f1": float(tf1), "history": hist},
              open(out / "metrics.json", "w"), indent=2)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(7), EMO_7, rotation=45, ha="right")
    plt.yticks(range(7), EMO_7)
    for i in range(7):
        for j in range(7):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.title(f"ResNet18 + FER2013 test  acc={tacc:.3f}  macroF1={tf1:.3f}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=130)

    print(f"[EXP1b DONE] test_acc={tacc:.4f} test_f1={tf1:.4f} -> {out}")


if __name__ == "__main__":
    main()
