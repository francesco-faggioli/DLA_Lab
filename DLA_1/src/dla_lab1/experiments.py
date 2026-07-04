from __future__ import annotations

from pathlib import Path

from sklearn.svm import SVC

from .config import experiment_config
from .data import build_dataloaders
from .features import extract_features, save_feature_cache
from .models import build_classifier, build_feature_extractor
from .paths import ensure_dir, resolve_path
from .seed import seed_everything
from .tracking import save_run_artifacts
from .train import configure_torch_for_hardware, resolve_device, train_model


def batch_size_for(config: dict, batch_size_key: str) -> int:
    """
    Serve a leggere il batch size dalla sezione hardware del config.

    In questo modo possiamo cambiare batch size in un solo punto, adattandolo
    alla memoria disponibile sulla GPU.

    Args:
        config: Configurazione completa del progetto.
        batch_size_key: Nome della chiave nella sezione `hardware`.

    Returns:
        Batch size intero da usare nella run.
    """
    return int(config["hardware"][batch_size_key])


def run_feature_svm(config: dict, root: str | Path = ".") -> dict:
    """
    Serve a lanciare la baseline ResNet feature extractor + SVM.

    E' la baseline stabile dell'esercizio precedente: estrae feature con
    ResNet-18 congelata e addestra una SVM lineare.

    Args:
        config: Configurazione caricata da `config.yaml`.
        root: Cartella radice del progetto, usata per risolvere path relativi.

    Returns:
        Dizionario con classificatore SVM, label test e predizioni.
    """
    seed_everything(int(config["project"]["seed"]))
    exp = experiment_config(config, "feature_svm")
    device = resolve_device(config["project"].get("device", "auto"))
    configure_torch_for_hardware(device, bool(config["hardware"].get("allow_tf32", True)))

    batch_size = batch_size_for(config, exp["experiment"].get("batch_size_key", "batch_size_feature_extraction"))
    loaders = build_dataloaders(
        data_root=resolve_path(config["paths"]["data_root"], root),
        image_size=int(config["dataset"]["image_size"]),
        batch_size=batch_size,
        val_split=float(config["dataset"]["val_split"]),
        track_size=int(config["dataset"]["track_size"]),
        seed=int(config["project"]["seed"]),
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
    )

    model_cfg = exp["model"]
    extractor = build_feature_extractor(model_cfg["name"], model_cfg.get("weights", "DEFAULT"))
    train_features, train_labels = extract_features(extractor, loaders["train"], device)
    test_features, test_labels = extract_features(extractor, loaders["test"], device)

    artifact_dir = ensure_dir(resolve_path(config["paths"]["artifacts_dir"], root) / "features")
    save_feature_cache(
        artifact_dir / "resnet18_gtsrb_features_64x64.pt",
        train_features=train_features,
        train_labels=train_labels,
        test_features=test_features,
        test_labels=test_labels,
    )

    svm_cfg = config.get("svm", {})
    classifier = SVC(
        kernel=svm_cfg.get("kernel", "linear"),
        C=float(svm_cfg.get("C", 1.0)),
        cache_size=float(svm_cfg.get("cache_size", 2000)),
    )
    classifier.fit(train_features.numpy(), train_labels.numpy())
    predictions = classifier.predict(test_features.numpy())
    return {"classifier": classifier, "test_labels": test_labels.numpy(), "predictions": predictions}


def run_finetuning(config: dict, experiment_name: str, root: str | Path = ".") -> dict:
    """
    Serve a lanciare un esperimento di fine-tuning definito nel config.

    Costruisce DataLoader, modello, loss, optimizer e training loop partendo
    da `config.yaml`, cosi' il notebook resta piu' pulito e riproducibile.

    Args:
        config: Configurazione generale del progetto.
        experiment_name: Nome dell'esperimento definito in `config.yaml`.
        root: Cartella radice del progetto.

    Returns:
        Dizionario con modello addestrato, history, DataLoader, device e artifact salvati.
    """
    seed_everything(int(config["project"]["seed"]))
    exp = experiment_config(config, experiment_name)
    device = resolve_device(config["project"].get("device", "auto"))
    configure_torch_for_hardware(device, bool(config["hardware"].get("allow_tf32", True)))

    batch_size = batch_size_for(config, exp["experiment"].get("batch_size_key", "batch_size_finetune_frozen"))
    augmentation = exp["experiment"].get("augmentation", "none")
    loaders = build_dataloaders(
        data_root=resolve_path(config["paths"]["data_root"], root),
        image_size=int(config["dataset"]["image_size"]),
        batch_size=batch_size,
        val_split=float(config["dataset"]["val_split"]),
        track_size=int(config["dataset"]["track_size"]),
        seed=int(config["project"]["seed"]),
        num_workers=int(config["dataset"]["num_workers"]),
        pin_memory=bool(config["dataset"]["pin_memory"]),
        augmentation=augmentation,
    )

    model_cfg = exp["model"]
    model = build_classifier(
        model_name=model_cfg["name"],
        num_classes=int(config["project"]["num_classes"]),
        weights=model_cfg.get("weights", "DEFAULT"),
        freeze_backbone=bool(model_cfg.get("freeze_backbone", True)),
        unfreeze_layer4=bool(model_cfg.get("unfreeze_layer4", False)),
    )
    checkpoint_dir = ensure_dir(resolve_path(config["paths"]["checkpoints_dir"], root))
    model, history = train_model(
        model,
        loaders["train"],
        loaders["val"],
        device,
        exp,
        class_weights=loaders["class_weights"],
        checkpoint_path=checkpoint_dir / f"{experiment_name}.pt",
    )
    artifacts = save_run_artifacts(exp, experiment_name, history, root=root)
    return {"model": model, "history": history, "loaders": loaders, "device": device, "artifacts": artifacts}
