"""Experiment 2: ResNet18 on FERPlus (8-class soft labels).

Same data as EfficientNet-B0, different architecture -- isolates the effect of
architecture choice. Uses FERPlus soft labels with soft cross-entropy loss.
"""
import json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import cv2, random

EMO_8 = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]


# -------------------------------------------------------------------
# FERPlus Dataset -- RGB output for ResNet
# -------------------------------------------------------------------
class FERPlusRGBDataset(Dataset):
    """FERPlus dataset that outputs 3-channel RGB for ResNet/VGG.
    Same augmentations as FERPlusDataset, but:
    - Grayscale repeat 3x for RGB
    - Resize to 224x224
    - ImageNet normalization
    """
    def __init__(self, df, img_size=224, train=True, degrade_p=0.5):
        self.pix = df["pixels"].tolist()
        self.soft = np.asarray(df["soft"].tolist(), dtype=np.float32)
        self.label = df["label"].to_numpy().astype(np.int64)
        self.img_size = img_size
        self.train = train
        self.degrade_p = degrade_p

    def __len__(self):
        return len(self.pix)

    def _img(self, i):
        return np.array(self.pix[i].split(), dtype=np.uint8).reshape(48, 48)

    def _degrade(self, img):
        if random.random() < self.degrade_p:
            s = random.choice([12, 16, 20, 24, 32])
            img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA),
                             (48, 48), interpolation=cv2.INTER_LINEAR)
        if random.random() < self.degrade_p * 0.6:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        return img

    def _augment(self, img):
        img = self._degrade(img)
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
        # Grayscale -> RGB (repeat 3x)
        img = np.stack([img, img, img], axis=-1)
        x = torch.from_numpy(img).float().permute(2, 0, 1).div_(255.0)
        # ImageNet normalization
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return x, torch.from_numpy(self.soft[i]), int(self.label[i])


# -------------------------------------------------------------------
# Soft cross-entropy loss
# -------------------------------------------------------------------
def soft_ce(logits, soft):
    return -(soft * torch.log_softmax(logits, 1)).sum(1).mean()


# -------------------------------------------------------------------
# Data loaders
# -------------------------------------------------------------------
def make_loaders(parquet_path, img_size, bs, nw):
    df = pd.read_parquet(parquet_path)
    tr = df[df.split == "train"]
    va = df[df.split == "val"]
    te = df[df.split == "test"]

    tr_ds = FERPlusRGBDataset(tr, img_size, train=True)
    va_ds = FERPlusRGBDataset(va, img_size, train=False)
    te_ds = FERPlusRGBDataset(te, img_size, train=False)

    counts = np.bincount(tr["label"].to_numpy(), minlength=8)
    w = 1.0 / np.clip(counts.astype(np.float64), 1, None)
    sw = w[tr["label"].to_numpy()]
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True)

    return (DataLoader(tr_ds, bs, sampler=sampler, num_workers=nw, pin_memory=True, drop_last=True),
            DataLoader(va_ds, bs, shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(te_ds, bs, shuffle=False, num_workers=nw, pin_memory=True))


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, soft, y in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy())
        ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    acc = (y == p).mean()
    f1 = f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p, labels=list(range(8)))
    return acc, f1, cm


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ferplus.parquet")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}  {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    tl, vl, te = make_loaders(a.data, a.img_size, a.bs, a.workers)

    model = timm.create_model("resnet18", pretrained=True, num_classes=8, in_chans=3).to(device)
    print(f"[model] ResNet18  params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    out = Path("runs") / f"resnet18_ferplus_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    best, hist = 0.0, []

    for ep in range(a.epochs):
        model.train()
        run, tot = 0.0, 0
        for x, soft, y in tl:
            x, soft = x.to(device), soft.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=(device.type == "cuda")):
                loss = soft_ce(model(x), soft)
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
            torch.save({"model": model.state_dict(), "img_size": a.img_size, "args": vars(a)}, out / "best.pt")

    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    tacc, tf1, cm = evaluate(model, te, device)
    json.dump({"val_best": float(best), "test_acc": float(tacc), "test_f1": float(tf1), "history": hist},
              open(out / "metrics.json", "w"), indent=2)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(8), EMO_8, rotation=45, ha="right")
    plt.yticks(range(8), EMO_8)
    for i in range(8):
        for j in range(8):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
    plt.title(f"ResNet18 + FERPlus test  acc={tacc:.3f}  macroF1={tf1:.3f}")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=130)

    print(f"[EXP2 DONE] test_acc={tacc:.4f} test_f1={tf1:.4f} -> {out}")


if __name__ == "__main__":
    main()
