import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision Training per GPU NVIDIA
import wandb  # Weights & Biases per logging esperimenti
from tqdm import tqdm  # Barre di progresso
import os

class FocalLoss(nn.Module):
    """
    Focal Loss ottimizzata per il class imbalance.
    
    La Focal Loss è una variante della Cross Entropy che penalizza gli esempi facili
    concentrandosi sugli esempi difficili da classificare. Particolarmente utile per
    dataset fortemente sbilanciati come GTSRB.
    
    Formula: FL = -alpha * (1 - pt)^gamma * log(pt)
    dove:
    - pt = probabilità predetta per la classe corretta
    - gamma = parametro di focalizzazione (penalizza esempi facili)
    - alpha = parametro di bilanciamento (opzionale)
    
    I pesi vengono gestiti come buffer per evitare conversioni inutili durante il forward.
    
    Args:
        alpha (float): Parametro di pesamento. Default: 1.0 (neutro)
            - Se alpha=1.0: nessun bilanciamento aggiuntivo
            - Se alpha≠1.0: bilancia classi positive/negative (tipico in object detection)
            - NOTA: Se usi 'weights', lascia alpha=1.0 per evitare doppio bilanciamento
        gamma (float): Parametro di focalizzazione. Default: 2.0
            - Penalizza gli esempi facili (quelli con alta probabilità)
            - Valori tipici: 0-5 (2.0 è lo standard dalla letteratura)
            - Gamma più alto = più focus sugli esempi difficili
        weights (torch.Tensor, optional): Pesi per le singole classi. Default: None
            - Usa questo per bilanciare classi sbilanciate numericamente
            - Calcolati con compute_class_weight('balanced', ...)
        reduction (str): Tipo di riduzione ('mean', 'sum'). Default: 'mean'
    """
    def __init__(self, alpha=1.0, gamma=2.0, weights=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        # Se vengono forniti i pesi delle classi, li registriamo come buffer
        # register_buffer: salva il tensor nello state_dict ma NON nei parametri trainabili
        # Automaticamente spostato sul device corretto insieme al modello
        if weights is not None:
            # IMPORTANTE: Convertiamo numpy array → tensor PRIMA di registrare
            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            self.register_buffer('class_weights', weights_tensor)
        else:
            self.class_weights = None

    def forward(self, inputs, targets):
        """
        Forward pass della Focal Loss.
        
        Args:
            inputs (torch.Tensor): Logits del modello (output pre-softmax)
                                  shape: (batch_size, num_classes)
            targets (torch.Tensor): Target labels (indici delle classi corrette)
                                   shape: (batch_size,)
            
        Returns:
            torch.Tensor: Focal loss value (scalare se reduction='mean')
        """
        # Step 1: Calcola Cross Entropy standard senza riduzione
        # reduction='none' ritorna un tensor con una loss per ogni sample
        # weight=self.class_weights applica i pesi di bilanciamento
        ce_loss = F.cross_entropy(
            inputs, targets, 
            reduction='none',  # Non ridurre subito (serve per calcolare pt)
            weight=self.class_weights  # Buffer già sul device corretto
        )
        
        # Step 2: Calcola pt (probabilità della classe corretta)
        # exp(-ce_loss) = probabilità predetta per la classe target
        pt = torch.exp(-ce_loss)
        
        # Step 3: Applica la formula della Focal Loss
        # (1 - pt)^gamma penalizza esempi facili (pt alto)
        # Se pt = 0.9 (esempio facile) e gamma=2 → (1-0.9)^2 = 0.01 (loss molto ridotta)
        # Se pt = 0.3 (esempio difficile) e gamma=2 → (1-0.3)^2 = 0.49 (loss alta)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Step 4: Applica la riduzione (media, somma, o nessuna)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_and_evaluate(config, model, train_loader, val_loader, device, class_weights=None):
    """
    Complete training and validation loop.

    Supports:
    - CrossEntropy, WeightedCrossEntropy, FocalLoss
    - Adam, AdamW, SGD
    - StepLR scheduler
    - early stopping on validation accuracy
    - optional mixed precision
    - optional Weights & Biases logging
    - configurable checkpoint path
    """

    # ------------------------------------------------------------
    # Optional Weights & Biases setup
    # ------------------------------------------------------------
    use_wandb = config.get("use_wandb", True)

    if use_wandb:
        run = wandb.init(
            project=config.get("wandb_project", "DLA_Lab1"),
            name=config["experiment_name"],
            config=config,
            reinit=True
        )
    else:
        run = None

    # ------------------------------------------------------------
    # Mixed precision setup
    # ------------------------------------------------------------
    use_amp = config.get("use_amp", device.type == "cuda")
    scaler = GradScaler() if use_amp and device.type == "cuda" else None

    if scaler is not None:
        print("⚡ Mixed Precision Training enabled")

    # ------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------
    if config["loss"] == "CrossEntropy":
        criterion = nn.CrossEntropyLoss()

    elif config["loss"] == "WeightedCrossEntropy":
        if class_weights is None:
            raise ValueError("WeightedCrossEntropy requires class_weights.")

        weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        assert len(weights_tensor) == 43, f"Expected 43 weights, got {len(weights_tensor)}"
        criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    elif config["loss"] == "FocalLoss":
        if class_weights is None:
            raise ValueError("FocalLoss requires class_weights.")

        weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        assert len(weights_tensor) == 43, f"Expected 43 weights, got {len(weights_tensor)}"

        criterion = FocalLoss(
            alpha=config.get("alpha_focal", 1.0),
            gamma=config.get("gamma_focal", 2.0),
            weights=weights_tensor
        ).to(device)

    else:
        raise ValueError(
            f"Unsupported loss: {config['loss']}. "
            "Choose from CrossEntropy, WeightedCrossEntropy, FocalLoss."
        )

    # ------------------------------------------------------------
    # Optimizer: only train parameters with requires_grad=True
    # ------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check requires_grad settings.")

    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            trainable_params,
            lr=config["learning_rate"]
        )

    elif config["optimizer"] == "AdamW":
        optimizer = optim.AdamW(
            trainable_params,
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.01)
        )

    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            trainable_params,
            lr=config["learning_rate"],
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 1e-4)
        )

    else:
        raise ValueError(
            f"Unsupported optimizer: {config['optimizer']}. "
            "Choose from Adam, AdamW, SGD."
        )

    # ------------------------------------------------------------
    # Scheduler and checkpoint
    # ------------------------------------------------------------
    scheduler = StepLR(
        optimizer,
        step_size=config.get("step_size", 4),
        gamma=config.get("gamma_lr", 0.1)
    )

    checkpoint_path = config.get(
        "checkpoint_path",
        os.path.join("checkpoints", f"{config['experiment_name']}.pt")
    )

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0
    patience = config.get("patience", 5)
    patience_counter = 0
    max_grad_norm = config.get("max_grad_norm", 1.0)

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    for epoch in range(1, config["epochs"] + 1):

        model.train()

        running_train_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']} [TRAIN]")

        for inputs, labels in pbar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                scaler.step(optimizer)
                scaler.update()

            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            total_train += labels.size(0)
            correct_train += (preds == labels).sum().item()

        train_loss = running_train_loss / total_train
        train_acc = correct_train / total_train

        # --------------------------------------------------------
        # Validation loop
        # --------------------------------------------------------
        model.eval()

        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.inference_mode():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{config['epochs']} [VAL]")

            for inputs, labels in pbar_val:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

                preds = outputs.argmax(dim=1)
                total_val += labels.size(0)
                correct_val += (preds == labels).sum().item()

        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "Train/Loss": train_loss,
                "Train/Accuracy": train_acc,
                "Validation/Loss": val_loss,
                "Validation/Accuracy": val_acc,
                "Learning_Rate": current_lr
            })

        # --------------------------------------------------------
        # Early stopping
        # --------------------------------------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ New best model saved: val_acc = {val_acc:.4f}")

        else:
            patience_counter += 1
            print(f"  ⚠️ No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        scheduler.step()

    # ------------------------------------------------------------
    # Load best model
    # ------------------------------------------------------------
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ Best model loaded from {checkpoint_path}")

    if use_wandb:
        wandb.finish()

    print("✓ Training completed")

    return model