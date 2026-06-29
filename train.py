import json, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from dataset import FERPlusDataset, EMO

def soft_ce(logits, soft):
    return -(soft * torch.log_softmax(logits, 1)).sum(1).mean()

def make_loaders(df, img_size, bs, nw):
    tr, va, te = df[df.split=="train"], df[df.split=="val"], df[df.split=="test"]
    tr_ds = FERPlusDataset(tr, img_size, train=True)
    va_ds = FERPlusDataset(va, img_size, train=False)
    te_ds = FERPlusDataset(te, img_size, train=False)
    counts = np.bincount(tr["label"].to_numpy(), minlength=8)
    w = 1.0 / np.clip(counts, 1, None)
    sw = w[tr["label"].to_numpy()]
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double), len(sw), replacement=True)
    return (DataLoader(tr_ds, bs, sampler=sampler, num_workers=nw, pin_memory=True, drop_last=True),
            DataLoader(va_ds, bs, shuffle=False, num_workers=nw, pin_memory=True),
            DataLoader(te_ds, bs, shuffle=False, num_workers=nw, pin_memory=True))

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); ys, ps = [], []
    for x, soft, y in loader:
        x = x.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out = model(x)
        ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return (y == p).mean(), f1_score(y, p, average="macro"), confusion_matrix(y, p, labels=list(range(8)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/ferplus.parquet")
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=2)   # Windows 安全值
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[train] device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "")
    df = pd.read_parquet(a.data)
    tl, vl, te = make_loaders(df, a.img_size, a.bs, a.workers)

    model = timm.create_model(a.model, pretrained=True, num_classes=8, in_chans=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    out = Path("runs") / f"{a.model}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    best, hist = 0.0, []

    for ep in range(a.epochs):
        model.train(); run = tot = 0
        for x, soft, y in tl:
            x, soft = x.to(device), soft.to(device)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=(device == "cuda")):
                loss = soft_ce(model(x), soft)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            run += loss.item() * x.size(0); tot += x.size(0)
        sched.step()
        vacc, vf1, _ = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_loss": run/tot, "val_acc": float(vacc), "val_f1": float(vf1)})
        print(f"ep{ep:02d} loss {run/tot:.3f}  val_acc {vacc:.4f}  val_f1 {vf1:.4f}")
        if vacc > best:
            best = vacc
            torch.save({"model": model.state_dict(), "args": vars(a)}, out/"best.pt")

    ckpt = torch.load(out/"best.pt"); model.load_state_dict(ckpt["model"])
    tacc, tf1, cm = evaluate(model, te, device)
    json.dump({"val_best": float(best), "test_acc": float(tacc), "test_f1": float(tf1), "history": hist},
              open(out/"metrics.json", "w"), indent=2)

    plt.figure(figsize=(7, 6)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(8), EMO, rotation=45, ha="right"); plt.yticks(range(8), EMO)
    for i in range(8):
        for j in range(8):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8)
    plt.title(f"FERPlus test  acc={tacc:.3f}  macroF1={tf1:.3f}")
    plt.tight_layout(); plt.savefig(out/"confusion_matrix.png", dpi=130)

    print(f"[TRAIN DONE] test_acc={tacc:.4f} test_f1={tf1:.4f} -> {out}")

if __name__ == "__main__":   # Windows 多进程 DataLoader 必须有这个守卫
    main()
