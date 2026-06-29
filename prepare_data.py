"""拉取 FER2013 + FERPlus 标签，合并过滤，存 data/ferplus.parquet"""
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("data"); DATA.mkdir(exist_ok=True)
FER = DATA / "fer2013.csv"
NEW = DATA / "fer2013new.csv"
OUT = DATA / "ferplus.parquet"
EMO = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

def ensure_ferplus_labels():
    if NEW.exists():
        return
    url = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
    print("[data] downloading FERPlus labels:", url)
    urllib.request.urlretrieve(url, NEW)

def ensure_fer2013():
    if FER.exists():
        return
    # Try HuggingFace mirror first (datasets library)
    print("[data] fer2013.csv missing — trying HuggingFace mirror...")
    try:
        import sys, subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset
        ds = load_dataset("fer2013", trust_remote_code=True)
        # ds has splits: train, test
        rows = []
        for split_name, split_ds in ds.items():
            usage_map = {"train": "Training", "test": "PrivateTest"}
            usage = usage_map.get(split_name, split_name)
            for item in split_ds:
                pixels = " ".join(str(int(v)) for v in item["pixels"].flatten())
                rows.append({"emotion": item["label"], "pixels": pixels, "Usage": usage})
        df = pd.DataFrame(rows)
        # FER2013 via HF datasets gives 35887 rows
        print(f"[data] HuggingFace fer2013: {len(df)} rows")
        if len(df) == 35887:
            df.to_csv(FER, index=False)
            print("[data] saved fer2013.csv from HuggingFace mirror")
            return
        else:
            print(f"[data] Warning: expected 35887 rows, got {len(df)} — trying Kaggle...")
    except Exception as e:
        print(f"[data] HuggingFace failed: {e}")
        print("[data] Trying Kaggle CLI...")

    # Try Kaggle CLI
    try:
        import subprocess, sys
        subprocess.check_call(["kaggle", "datasets", "download", "-d", "deadskull7/fer2013", "-p", str(DATA), "--unzip"])
        if FER.exists():
            print("[data] saved fer2013.csv from Kaggle")
            return
    except Exception as e:
        print(f"[data] Kaggle failed: {e}")

    # Try alternative Kaggle dataset
    try:
        import subprocess
        subprocess.check_call(["kaggle", "datasets", "download", "-d", "msambare/fer2013", "-p", str(DATA), "--unzip"])
        if FER.exists():
            print("[data] saved fer2013.csv from Kaggle (msambare)")
            return
    except Exception as e:
        print(f"[data] Kaggle alternative failed: {e}")

    raise SystemExit("fer2013.csv 缺失 —— 见 §9 获取方式。")

def main():
    ensure_ferplus_labels()
    ensure_fer2013()
    fer = pd.read_csv(FER)
    new = pd.read_csv(NEW)
    new.columns = [c.strip() for c in new.columns]
    assert len(fer) == len(new), f"行数不一致 {len(fer)} vs {len(new)}"

    # FERPlus 列(去空格后): Usage, Image name, neutral..contempt, unknown, NF
    usage_col = [c for c in new.columns if c.lower() == "usage"][0]
    vote_cols = EMO + ["unknown", "NF"]
    votes = new[vote_cols].fillna(0).to_numpy(dtype=np.float32)   # (N,10)

    arg = votes.argmax(1)
    keep = (arg < 8) & (votes.sum(1) > 0)        # 丢掉 unknown/NF 主导 & 全空
    emo = votes[:, :8]
    s = emo.sum(1, keepdims=True); s[s == 0] = 1
    soft = emo / s
    label = emo.argmax(1)

    usage = new[usage_col].astype(str).str.strip().to_numpy()
    split = np.where(usage == "Training", "train",
            np.where(usage == "PublicTest", "val", "test"))

    df = pd.DataFrame({
        "pixels": fer["pixels"].values,
        "label": label.astype(np.int64),
        "split": split,
        "soft": [r.tolist() for r in soft],
    })[keep].reset_index(drop=True)

    df.to_parquet(OUT)
    print("[data] saved", OUT, "rows:", len(df))
    print(df.groupby("split").size())
    tr = df[df.split == "train"]
    print("train 各类:")
    print(pd.Series(tr["label"]).map(dict(enumerate(EMO))).value_counts())
    print("[STAGE DATA OK]")

if __name__ == "__main__":
    main()
