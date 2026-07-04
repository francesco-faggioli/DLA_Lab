from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm.auto import tqdm

from .losses import build_loss
from .paths import ensure_dir
from .tracking import finish_wandb_run, init_wandb_run, log_checkpoint_to_wandb, log_epoch_to_wandb


@dataclass
class EpochMetrics:
    """
    Serve a salvare le metriche principali di una singola epoca.

    La convertiamo poi in tabella per confrontare loss e accuracy di train
    e validation durante il fine-tuning.
    """

    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    learning_rate: float


def resolve_device(device: str = "auto") -> torch.device:
    """
    Serve a scegliere dove eseguire il training.

    Con `auto` usa CUDA se disponibile, altrimenti MPS su Mac o CPU.
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def configure_torch_for_hardware(device: torch.device, allow_tf32: bool = True) -> None:
    """
    Serve ad attivare impostazioni PyTorch utili per il PC disponibile.

    Su GPU NVIDIA abilita benchmark cuDNN e TF32, che di solito velocizzano
    il training senza cambiare la logica dell'esperimento.
    """
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32


def build_optimizer(name: str, params, learning_rate: float, weight_decay: float = 0.0, momentum: float = 0.9):
    """
    Serve a creare l'optimizer scelto nella configurazione.

    In questo modo possiamo provare Adam, AdamW o SGD senza riscrivere
    il training loop nel notebook.
    """
    if name == "Adam":
        return optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "AdamW":
        return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "SGD":
        return optim.SGD(params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer {name!r}.")


def build_scheduler(name: str, optimizer, epochs: int, step_size: int = 4, gamma: float = 0.1, min_lr: float = 1e-6):
    """
    Serve a modificare il learning rate durante il training.

    `step` replica la logica usata negli esperimenti vecchi, mentre `cosine`
    e' utile per run piu' lunghe e progressive.
    """
    if name == "none":
        return None
    if name == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=min_lr)
    raise ValueError(f"Unsupported scheduler {name!r}.")


def run_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    optimizer=None,
    scaler=None,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """
    Serve a eseguire una singola epoca di train oppure di valutazione.

    Se riceve un optimizer aggiorna i pesi del modello; se l'optimizer manca,
    calcola solo loss e accuracy senza modificare il modello.
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    context = torch.enable_grad() if is_train else torch.inference_mode()
    with context:
        for inputs, labels in tqdm(dataloader, leave=False):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            use_amp = scaler is not None and device.type == "cuda"
            autocast_context = torch.autocast(device_type="cuda") if use_amp else nullcontext()
            with autocast_context:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, total_correct / total


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    config: dict[str, Any],
    class_weights: torch.Tensor | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[nn.Module, list[EpochMetrics]]:
    """
    Serve ad addestrare un modello e conservare il miglior checkpoint.

    Monitora la validation accuracy, salva il modello migliore e applica
    early stopping se la validation non migliora per troppe epoche.
    """
    training = config.get("training", config)
    model = model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found.")

    criterion = build_loss(training.get("loss", "CrossEntropy"), class_weights=class_weights, device=device, **training)
    optimizer = build_optimizer(
        training.get("optimizer", "AdamW"),
        trainable_params,
        learning_rate=float(training.get("learning_rate", 5e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        momentum=float(training.get("momentum", 0.9)),
    )
    scheduler = build_scheduler(
        training.get("scheduler", "cosine"),
        optimizer,
        epochs=int(training.get("epochs", 20)),
        step_size=int(training.get("step_size", 4)),
        gamma=float(training.get("gamma_lr", 0.1)),
        min_lr=float(training.get("min_lr", 1e-6)),
    )

    use_amp = bool(training.get("use_amp", config.get("hardware", {}).get("use_amp", True))) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    wandb_run = init_wandb_run(config)

    if checkpoint_path is None:
        checkpoint_dir = ensure_dir(config.get("paths", {}).get("checkpoints_dir", "checkpoints"))
        checkpoint_path = checkpoint_dir / f"{config.get('experiment_name', 'model')}.pt"
    else:
        checkpoint_path = Path(checkpoint_path)
        ensure_dir(checkpoint_path.parent)

    best_val_acc = -1.0
    patience = int(training.get("patience", 5))
    patience_counter = 0
    history: list[EpochMetrics] = []

    for epoch in range(1, int(training.get("epochs", 20)) + 1):
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        history.append(EpochMetrics(epoch, train_loss, train_acc, val_loss, val_acc, lr))

        print(
            f"Epoch {epoch:02d} | lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        log_epoch_to_wandb(
            wandb_run,
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "validation/loss": val_loss,
                "validation/accuracy": val_acc,
                "learning_rate": lr,
            },
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if scheduler is not None:
            scheduler.step()

    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if wandb_run is not None:
            wandb_run.summary["best_val_acc"] = best_val_acc
            log_checkpoint_to_wandb(
                wandb_run,
                checkpoint_path,
                artifact_name=f"{config.get('experiment_name', 'model')}-best-checkpoint",
            )
    finish_wandb_run(wandb_run)
    return model, history
