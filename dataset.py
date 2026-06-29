import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

EMO = ["neutral","happiness","surprise","sadness","anger","disgust","fear","contempt"]

class FERPlusDataset(Dataset):
    def __init__(self, df, img_size=128, train=True, degrade_p=0.5):
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
        if random.random() < self.degrade_p:               # 随机降分辨率再放大
            s = random.choice([12, 16, 20, 24, 32])
            img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA),
                             (48, 48), interpolation=cv2.INTER_LINEAR)
        if random.random() < self.degrade_p * 0.6:          # 随机高斯模糊
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
        if random.random() < 0.5:                            # 亮度/对比
            a, b = random.uniform(0.8, 1.2), random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        return img

    def __getitem__(self, i):
        img = self._img(i)
        if self.train:
            img = self._augment(img)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(img).float().div_(255.0).sub_(0.5).div_(0.5).unsqueeze(0)  # (1,H,W)
        return x, torch.from_numpy(self.soft[i]), int(self.label[i])
