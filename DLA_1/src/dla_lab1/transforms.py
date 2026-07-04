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

    Args:
        image_size: Dimensione finale quadrata delle immagini.
        train: Se True, abilita eventuali augmentation per il training.
        augmentation: Tipo di augmentation: `none`, `aggressive`, `conservative` o `spatial`.

    Returns:
        Composizione torchvision di trasformazioni da applicare alle immagini.
    """
    resize_size = image_size + 6 if train and augmentation in {"aggressive", "conservative", "spatial"} else image_size
    steps = [
        T.Resize((resize_size, resize_size), antialias=True),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ]

    if train and augmentation == "aggressive":
        steps.insert(1, T.RandomCrop((image_size, image_size)))
        steps.insert(2, T.RandomHorizontalFlip(p=0.3))
        steps.insert(3, T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05))
        steps.insert(4, T.RandomRotation(degrees=10))
    elif train and augmentation == "conservative":
        steps.insert(1, T.RandomCrop((image_size, image_size)))
        steps.insert(2, T.ColorJitter(brightness=0.1, contrast=0.1))
        steps.insert(3, T.RandomRotation(degrees=3))
    elif train and augmentation == "spatial":
        steps.insert(1, T.RandomCrop((image_size, image_size)))
        steps.insert(2, T.RandomRotation(degrees=5))

    steps.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return T.Compose(steps)
