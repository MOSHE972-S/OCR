import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

VOCAB = (
    ["<blank>"]
    + list("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
    + list(" 0123456789.,!?:;\'\"-/()")
)
CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
NUM_CLASSES = len(VOCAB)

WIDTH_MAP = {"word": 128, "line": 512, "sentence": 1024}


class HandwritingDataset(Dataset):
    def __init__(self, csv_path, input_type="word", is_train=True):
        try:
            df = pd.read_csv(csv_path)
            self.data = df[df["type"] == input_type].reset_index(drop=True)
        except Exception:
            self.data = pd.DataFrame(columns=["image_path", "text"])
        self.max_width = WIDTH_MAP.get(input_type, 128)
        self.is_train = is_train
        self.transform = (
            A.Compose([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.SafeRotate(limit=5, p=0.4),
                A.ElasticTransform(alpha=1, sigma=50, p=0.2),
                A.GridDistortion(p=0.2),
                A.RandomBrightnessContrast(p=0.3),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CoarseDropout(max_holes=5, max_height=8, max_width=8, p=0.15),
            ])
            if is_train else None
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row["text"])
        img = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((32, self.max_width), dtype=np.uint8)
        if self.transform:
            img = self.transform(image=img)["image"]
        h, w = img.shape
        new_w = min(int(w * 32.0 / h), self.max_width)
        img = cv2.resize(img, (new_w, 32))
        padded = np.full((32, self.max_width), 255, dtype=np.uint8)
        padded[:, :new_w] = img
        tensor = torch.FloatTensor(padded).unsqueeze(0) / 255.0
        labels = [CHAR2ID.get(c, 0) for c in text]
        return tensor, torch.LongTensor(labels), len(labels)


def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    return torch.stack(images), torch.cat(labels), torch.LongTensor(lengths)
