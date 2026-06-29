import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, torch, timm
from sklearn.metrics import classification_report
from dataset import FERPlusDataset, EMO
from torch.utils.data import DataLoader

def load_model(ckpt_path, model_name, device):
    m = timm.create_model(model_name, pretrained=False, num_classes=8, in_chans=1)
    m.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    return m.to(device).eval()

@torch.no_grad()
def run(model, loader, device):
    ys, ps = [], []
    for x, soft, y in loader:
        out = model(x.to(device))
        ps.append(out.argmax(1).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--data", default="data/ferplus.parquet")
    ap.add_argument("--img-size", type=int, default=128)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, a.model, device)
    df = pd.read_parquet(a.data)
    te = DataLoader(FERPlusDataset(df[df.split=="test"], a.img_size, train=False), 128)
    y, p = run(model, te, device)
    print(classification_report(y, p, target_names=EMO, digits=3))

if __name__ == "__main__":
    main()
