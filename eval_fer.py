import pandas as pd
from pathlib import Path
import cv2
import numpy as np
from fer.fer import FER
from sklearn.metrics import confusion_matrix, classification_report
import traceback
import gc

EMO = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]
EMO_FER_MAP = {
    "neutral": "neutral", "happy": "happiness", "surprise": "surprise",
    "sad": "sadness", "angry": "anger", "disgust": "disgust", "fear": "fear"
}

def load_all_self_data(self_dir):
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

    if not rows: return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

def resolve_image_path(filepath, self_dir="self"):
    fp = filepath.replace("\\", "/")
    if fp.startswith("raw/"):
        fp_stripped = fp[len("raw/"):]
    else:
        fp_stripped = fp
    p = Path(self_dir) / fp_stripped
    if p.exists():
        return p
    p2 = Path("user2_data") / fp
    if p2.exists():
        return p2
    return p

def main():
    print("Loading data...")
    df = load_all_self_data("self")
    if df.empty:
        print("No data found.")
        return
        
    print(f"Loaded {len(df)} images.")
    detector = FER(mtcnn=True)
    
    y_true = []
    y_pred = []
    
    for i, row in df.iterrows():
        path = resolve_image_path(row["filepath"])
        if not path.exists():
            continue
            
        img = cv2.imread(str(path))
        if img is None:
            continue
            
        true_label = row["final_label"]
        res = detector.detect_emotions(img)
        if res:
            best = max(res[0]["emotions"], key=res[0]["emotions"].get)
            pred_label = EMO_FER_MAP.get(best, "neutral")
        else:
            pred_label = "neutral"
            
        y_true.append(true_label)
        y_pred.append(pred_label)
        
        del img
        del res
        if i % 50 == 0:
            gc.collect()
        
        if (i+1) % 20 == 0:
            print(f"Processed {i+1}/{len(df)}")
            
    print("\n" + "="*50)
    print("Classification Report:")
    print(classification_report(y_true, y_pred, labels=EMO))
    
    cm = confusion_matrix(y_true, y_pred, labels=EMO)
    
    print("\nConfusion Matrix Table Data:")
    print("true \\ pred | neu | hap | sur | sad | ang | dis | fea | recall")
    for i, row in enumerate(cm):
        total = np.sum(row)
        recall = (row[i] / total * 100) if total > 0 else 0
        r_str = f"{row[0]:3d} | {row[1]:3d} | {row[2]:3d} | {row[3]:3d} | {row[4]:3d} | {row[5]:3d} | {row[6]:3d} | {recall:5.1f}%"
        print(f"{EMO[i]:9s} | {r_str}")
        
    correct = np.sum(np.diag(cm))
    total = np.sum(cm)
    overall = (correct / total * 100) if total > 0 else 0
    print(f"overall {total} accuracy {overall:.1f}%")

if __name__ == '__main__':
    main()
