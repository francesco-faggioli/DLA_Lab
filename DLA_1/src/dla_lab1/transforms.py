from __future__ import annotations

import torch
import torchvision.transforms.v2 as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int = 64, train: bool = False, augmentation: str = "none"):
    """
    Serve a costruire il preprocessing delle immagini.

    Per la baseline usa resize, conversione a tensore e normalizzazione ImageNet,
    cioe' le trasformazioni coerenti con una ResNet pre-addestrata.
    """
    steps = [
        T.Resize((image_size, image_size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]

    if train and augmentation == "conservative":
        steps.insert(1, T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)))
        steps.insert(2, T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10))
    elif train and augmentation == "spatial":
        steps.insert(1, T.RandomAffine(degrees=12, translate=(0.06, 0.06), scale=(0.92, 1.08)))

    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(steps)
