from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_TO_IDX: Dict[str, int] = {
    "dustin": 0,
    "eleven": 1,
    "hopper": 2,
    "lucas": 3,
    "steve": 4,
}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def default_transforms(split: str = "train") -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.ColorJitter(brightness=0.3, hue=0.1, p=0.5),
            A.GaussNoise(p=0.3),
            A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


class FinalisedFacesDataset(Dataset):
    """PyTorch Dataset for finalised_dataset/splits/{split}/{class}/

    Returns (image_tensor, label_int, filepath)
    """

    def __init__(self, root: str = "finalised_dataset/splits", split: str = "train", transform: Optional[A.Compose] = None):
        assert split in ("train", "val", "test"), "split must be 'train', 'val' or 'test'"
        self.root = Path(root)
        self.split = split
        self.class_to_idx = CLASS_TO_IDX
        self.samples: List[Tuple[Path, int]] = []

        for class_name, idx in self.class_to_idx.items():
            class_dir = self.root / split / class_name
            if not class_dir.exists():
                continue
            for p in sorted(class_dir.iterdir()):
                if p.is_file() and is_image_file(p):
                    self.samples.append((p, idx))

        if transform is None:
            self.transform = default_transforms(split)
        else:
            self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]

        # Load image as RGB numpy array for Albumentations
        with Image.open(path) as img:
            img_rgb = img.convert("RGB")
            img_np = np.array(img_rgb)

        augmented = self.transform(image=img_np)
        image_tensor = augmented["image"]

        return image_tensor, int(label), str(path)

    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {k: 0 for k in self.class_to_idx}
        for _p, label in self.samples:
            # reverse lookup
            for name, idx in self.class_to_idx.items():
                if idx == label:
                    counts[name] += 1
                    break
        return counts
