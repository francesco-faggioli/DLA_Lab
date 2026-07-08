from __future__ import annotations

import inspect
import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATASET_ID = "cornell-movie-review-data/rotten_tomatoes"
DISTILBERT_ID = "distilbert/distilbert-base-uncased"


@dataclass(frozen=True)
class SplitMetrics:
    """Metriche principali per uno split di classificazione."""

    split: str
    accuracy: float
    f1: float
    precision: float
    recall: float


def _patch_datasets_fingerprinting_for_python314() -> None:
    """
    Applica una patch locale al fingerprinting di Hugging Face Datasets.

    Args:
        Nessun argomento.

    Returns:
        None. Modifica `datasets.fingerprint.Hasher.hash` in memoria.

    Notes:
        Nell'ambiente `DLA2026-transformers` viene usato Python 3.14. Alcune
        versioni di `datasets`/`dill` non sono compatibili con la firma interna
        di `pickle._Pickler._batch_setitems` di Python 3.14. L'errore compare
        quando `datasets` prova a calcolare fingerprint e cache hash, non quando
        legge davvero i dati.

        La patch mantiene il comportamento originale quando funziona. Solo se
        compare l'errore `Pickler._batch_setitems`, usa un hash stabile basato
        su `repr(value)`. Per questo laboratorio va bene: non stiamo costruendo
        una pipeline di caching complessa, stiamo solo caricando un dataset noto.
    """
    import datasets.fingerprint as fingerprint

    if getattr(fingerprint.Hasher, "_dla_lab2_python314_patch", False):
        return

    original_hash = fingerprint.Hasher.hash

    @classmethod
    def safe_hash(cls, value: Any) -> str:
        try:
            return original_hash(value)
        except TypeError as exc:
            if "Pickler._batch_setitems" not in str(exc):
                raise
            payload = f"{type(value)!r}:{repr(value)}".encode("utf-8", errors="replace")
            return hashlib.sha256(payload).hexdigest()

    fingerprint.Hasher.hash = safe_hash
    fingerprint.Hasher._dla_lab2_python314_patch = True


def _load_rotten_tomatoes_from_local_cache() -> Any:
    """
    Carica Rotten Tomatoes dalla cache Hugging Face locale.

    Args:
        Nessun argomento. La funzione cerca nella cache standard dell'utente.

    Returns:
        DatasetDict con split train, validation e test.

    Raises:
        FileNotFoundError: Se i file Arrow del dataset non sono presenti in cache.
    """
    _patch_datasets_fingerprinting_for_python314()

    from datasets import Dataset, DatasetDict

    cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
    dataset_root = cache_root / "cornell-movie-review-data___rotten_tomatoes"
    candidates = sorted(dataset_root.glob("default/*/*/rotten_tomatoes-train.arrow"))
    if not candidates:
        raise FileNotFoundError(
            "Rotten Tomatoes is not available in the local Hugging Face cache. "
            "Fix the datasets/dill environment or download the dataset once."
        )

    split_dir = candidates[-1].parent
    split_files = {
        "train": split_dir / "rotten_tomatoes-train.arrow",
        "validation": split_dir / "rotten_tomatoes-validation.arrow",
        "test": split_dir / "rotten_tomatoes-test.arrow",
    }
    missing = [str(path) for path in split_files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing cached Rotten Tomatoes split files: {missing}")

    return DatasetDict({split: Dataset.from_file(str(path)) for split, path in split_files.items()})


def load_rotten_tomatoes(dataset_id: str = DATASET_ID) -> Any:
    """
    Scarica e carica il dataset Rotten Tomatoes da Hugging Face.

    Args:
        dataset_id: Identificativo Hugging Face del dataset.

    Returns:
        DatasetDict con gli split disponibili, di solito train, validation e test.

    Notes:
        In alcuni ambienti Windows/Python recenti `datasets` puo' fallire durante
        il controllo della cache con un errore `Pickler._batch_setitems`. Prima
        del caricamento applichiamo una patch locale al fingerprinting; se il
        download standard fallisce ancora, usiamo i file Arrow gia' presenti
        nella cache Hugging Face locale.
    """
    _patch_datasets_fingerprinting_for_python314()

    from datasets import load_dataset

    try:
        return load_dataset(dataset_id)
    except TypeError as exc:
        if "Pickler._batch_setitems" not in str(exc):
            raise
        return _load_rotten_tomatoes_from_local_cache()


def dataset_overview(ds_dict: Any) -> list[dict[str, Any]]:
    """
    Riassume dimensioni, colonne ed etichette degli split.

    Args:
        ds_dict: DatasetDict Hugging Face.

    Returns:
        Lista di dizionari, uno per split.
    """
    rows: list[dict[str, Any]] = []
    for split, dataset in ds_dict.items():
        rows.append(
            {
                "split": split,
                "num_rows": len(dataset),
                "columns": list(dataset.column_names),
                "labels": sorted(set(dataset["label"])),
            }
        )
    return rows


def sample_examples(dataset: Any, n: int = 8, seed: int = 42) -> list[dict[str, Any]]:
    """
    Estrae esempi casuali da uno split.

    Args:
        dataset: Split Hugging Face con colonne `text` e `label`.
        n: Numero di esempi da estrarre.
        seed: Seed per la selezione casuale.

    Returns:
        Lista di esempi con indice, testo ed etichetta.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)
    return [{"idx": int(i), "label": int(dataset[int(i)]["label"]), "text": dataset[int(i)]["text"]} for i in indices]


def load_distilbert_base(model_id: str = DISTILBERT_ID, device: str | None = None) -> tuple[Any, Any]:
    """
    Carica tokenizer e modello DistilBERT senza testa di classificazione.

    Args:
        model_id: Nome del modello Hugging Face.
        device: Device opzionale, per esempio `cuda` o `cpu`.

    Returns:
        Coppia `(tokenizer, model)`.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    if device is not None:
        model = model.to(device)
    model.eval()
    return tokenizer, model


def inspect_tokenizer_output(tokenizer: Any, texts: list[str], max_length: int = 128) -> Any:
    """
    Tokenizza alcuni testi per ispezionare input ids e attention mask.

    Args:
        tokenizer: Tokenizer Hugging Face.
        texts: Lista di frasi.
        max_length: Lunghezza massima usata per il truncation.

    Returns:
        BatchEncoding con `input_ids` e `attention_mask`.
    """
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")


def extract_cls_features_with_pipeline(
    texts: list[str],
    model: Any,
    tokenizer: Any,
    batch_size: int = 32,
    device: int = -1,
    truncation: bool = True,
) -> np.ndarray:
    """
    Estrae le feature del token CLS dall'ultimo layer di DistilBERT.

    Args:
        texts: Lista di stringhe da rappresentare.
        model: Modello DistilBERT senza testa di classificazione.
        tokenizer: Tokenizer associato al modello.
        batch_size: Dimensione del batch usata dalla pipeline.
        device: Indice GPU per Hugging Face pipeline, oppure -1 per CPU.
        truncation: Se True tronca sequenze troppo lunghe.

    Returns:
        Array NumPy di forma `(n_testi, hidden_size)`.
    """
    from transformers import pipeline

    extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer, device=device)
    outputs = extractor(texts, batch_size=batch_size, truncation=truncation, padding=True)

    features = []
    for item in outputs:
        arr = np.asarray(item)
        if arr.ndim == 3:
            features.append(arr[0, 0, :])
        elif arr.ndim == 2:
            features.append(arr[0, :])
        else:
            raise ValueError(f"Unexpected feature shape: {arr.shape}")
    return np.vstack(features)


def train_linear_svm(features: np.ndarray, labels: list[int] | np.ndarray, c: float = 1.0) -> Any:
    """
    Addestra una baseline SVM lineare sulle feature estratte.

    Args:
        features: Matrice `(n_esempi, n_feature)`.
        labels: Etichette binarie.
        c: Parametro di regolarizzazione della SVM.

    Returns:
        Pipeline Scikit-learn con StandardScaler e LinearSVC addestrata.
    """
    classifier = make_pipeline(
        StandardScaler(),
        LinearSVC(C=c, dual=False, max_iter=10_000, random_state=42),
    )
    classifier.fit(features, labels)
    return classifier


def evaluate_sklearn_classifier(classifier: Any, features: np.ndarray, labels: list[int] | np.ndarray, split: str) -> SplitMetrics:
    """
    Valuta un classificatore Scikit-learn.

    Args:
        classifier: Modello con metodo `predict`.
        features: Feature dello split da valutare.
        labels: Etichette vere.
        split: Nome dello split, per esempio `validation` o `test`.

    Returns:
        Oggetto SplitMetrics con accuracy, F1, precision e recall.
    """
    preds = classifier.predict(features)
    return SplitMetrics(
        split=split,
        accuracy=float(accuracy_score(labels, preds)),
        f1=float(f1_score(labels, preds, zero_division=0)),
        precision=float(precision_score(labels, preds, zero_division=0)),
        recall=float(recall_score(labels, preds, zero_division=0)),
    )


def classification_report_dict(classifier: Any, features: np.ndarray, labels: list[int] | np.ndarray) -> dict[str, Any]:
    """
    Produce il classification report di Scikit-learn in forma strutturata.

    Args:
        classifier: Modello con metodo `predict`.
        features: Feature dello split da valutare.
        labels: Etichette vere.

    Returns:
        Dizionario prodotto da `classification_report(..., output_dict=True)`.
    """
    preds = classifier.predict(features)
    return classification_report(labels, preds, output_dict=True, zero_division=0)


def tokenize_dataset_splits(ds_dict: Any, tokenizer: Any, max_length: int = 256) -> Any:
    """
    Tokenizza tutti gli split una sola volta con Dataset.map.

    Args:
        ds_dict: DatasetDict originale con colonna `text`.
        tokenizer: Tokenizer Hugging Face.
        max_length: Lunghezza massima delle sequenze.

    Returns:
        DatasetDict tokenizzato con formato PyTorch per input_ids, attention_mask e label.
    """

    def preprocess(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    tokenized = ds_dict.map(preprocess, batched=True)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "label"], output_all_columns=True)
    return tokenized


def load_sequence_classifier(model_id: str = DISTILBERT_ID, num_labels: int = 2) -> Any:
    """
    Carica DistilBERT con testa di classificazione.

    Args:
        model_id: Nome del checkpoint Hugging Face.
        num_labels: Numero di classi del task.

    Returns:
        Modello AutoModelForSequenceClassification pronto per il fine-tuning.
    """
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=num_labels)


def compute_sklearn_metrics(eval_pred: Any) -> dict[str, float]:
    """
    Calcola metriche per Hugging Face Trainer usando Scikit-learn.

    Args:
        eval_pred: Oggetto EvalPrediction oppure tupla `(logits, labels)`.

    Returns:
        Dizionario con accuracy, F1, precision e recall.
    """
    if hasattr(eval_pred, "predictions"):
        logits = eval_pred.predictions
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
    }


def build_training_arguments(
    output_dir: str | Path,
    learning_rate: float = 2e-5,
    train_batch_size: int = 32,
    eval_batch_size: int = 32,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    warmup_steps: int | None = None,
    metric_for_best_model: str = "accuracy",
) -> Any:
    """
    Costruisce TrainingArguments con default conservativi.

    Args:
        output_dir: Cartella in cui salvare checkpoint e log.
        learning_rate: Learning rate iniziale.
        train_batch_size: Batch size di training per device.
        eval_batch_size: Batch size di valutazione per device.
        num_train_epochs: Numero di epoche.
        weight_decay: Regolarizzazione AdamW.
        warmup_ratio: Frazione degli step usata per warmup, usata solo se
            `warmup_steps` non viene passato.
        warmup_steps: Numero esplicito di step di warmup. Preferito per evitare
            warning di deprecazione in Transformers 5.x.
        metric_for_best_model: Metrica usata per scegliere il checkpoint migliore.

    Returns:
        Istanza TrainingArguments.
    """
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "num_train_epochs": num_train_epochs,
        "weight_decay": weight_decay,
        "lr_scheduler_type": "cosine",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 50,
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": True,
        "report_to": "none",
    }
    if warmup_steps is not None:
        kwargs["warmup_steps"] = warmup_steps
    else:
        kwargs["warmup_ratio"] = warmup_ratio

    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def build_trainer(
    tokenizer: Any,
    training_args: Any,
    train_dataset: Any,
    eval_dataset: Any,
    model_init: Any | None = None,
    model: Any | None = None,
) -> Any:
    """
    Crea un Trainer standard per classificazione testuale.

    Args:
        tokenizer: Tokenizer usato per costruire il data collator.
        training_args: Istanza TrainingArguments.
        train_dataset: Split di training tokenizzato.
        eval_dataset: Split di validazione tokenizzato.
        model_init: Funzione opzionale che crea una nuova istanza del modello.
        model: Modello gia' istanziato, alternativo a `model_init`.

    Returns:
        Istanza Trainer pronta per `train()` ed `evaluate()`.

    Notes:
        Nell'ambiente notebook usato per il laboratorio, Transformers puo'
        aggiungere `NotebookProgressCallback`. In alcune versioni questo callback
        resta in uno stato non valido e fa fallire `evaluate()` con:
        `RuntimeError: on_train_begin must be called before on_evaluate`.
        Lo rimuoviamo subito dopo aver creato il Trainer; logging e metriche
        restano disponibili tramite i dizionari restituiti da `train/evaluate`.
    """
    from transformers import DataCollatorWithPadding, Trainer

    if model_init is None and model is None:
        raise ValueError("Pass either model_init or model.")

    kwargs = {
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_sklearn_metrics,
    }
    if model_init is not None:
        kwargs["model_init"] = model_init
    else:
        kwargs["model"] = model

    trainer = Trainer(**kwargs)
    disable_notebook_progress_callback(trainer)
    return trainer


def disable_notebook_progress_callback(trainer: Any) -> Any:
    """
    Rimuove il callback notebook di Transformers quando e' presente.

    Args:
        trainer: Istanza Hugging Face Trainer gia' creata.

    Returns:
        La stessa istanza `trainer`, modificata in-place.

    Notes:
        Serve anche se il Trainer e' stato creato prima di aggiornare
        `build_trainer()`: basta chiamare questa funzione prima di `evaluate()`.
    """
    try:
        from transformers.utils.notebook import NotebookProgressCallback

        trainer.remove_callback(NotebookProgressCallback)
    except Exception:
        pass
    return trainer


def suppress_subprocess_reader_unicode_errors() -> None:
    """
    Evita traceback rumorosi da thread interni che leggono subprocess output.

    Args:
        Nessun argomento.

    Returns:
        None. Installa un `threading.excepthook` locale e idempotente.

    Notes:
        In questo ambiente Windows/Python 3.14, alcune librerie importate da PEFT
        possono avviare controlli subprocess e fallire solo nel decoding del loro
        output (`UnicodeDecodeError` dentro `_readerthread`). Il training LoRA
        procede comunque. Sopprimiamo solo quel caso specifico per mantenere il
        notebook leggibile; qualsiasi altro errore di thread resta visibile.
    """
    if getattr(threading, "_dla_lab2_unicode_reader_patch", False):
        return

    original_hook = threading.excepthook

    def hook(args: threading.ExceptHookArgs) -> None:
        thread_name = getattr(args.thread, "name", "")
        if args.exc_type is UnicodeDecodeError and "_readerthread" in thread_name:
            return
        original_hook(args)

    threading.excepthook = hook
    threading._dla_lab2_unicode_reader_patch = True


def lora_sequence_classifier_init(
    model_id: str = DISTILBERT_ID,
    num_labels: int = 2,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
) -> Any:
    """
    Crea DistilBERT per sequence classification con adapter LoRA.

    Args:
        model_id: Checkpoint Hugging Face di partenza.
        num_labels: Numero di classi.
        rank: Rango LoRA.
        alpha: Scala LoRA.
        dropout: Dropout applicato ai layer LoRA.

    Returns:
        Modello PEFT con pochi parametri addestrabili.
    """
    suppress_subprocess_reader_unicode_errors()

    from peft import LoraConfig, TaskType, get_peft_model

    base_model = load_sequence_classifier(model_id=model_id, num_labels=num_labels)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_lin", "v_lin"],
        bias="none",
    )
    return get_peft_model(base_model, peft_config)


def partial_freezing_sequence_classifier_init(model_id: str = DISTILBERT_ID, num_labels: int = 2, frozen_layers: int = 4) -> Any:
    """
    Crea DistilBERT congelando embeddings e i primi layer transformer.

    Args:
        model_id: Checkpoint Hugging Face di partenza.
        num_labels: Numero di classi.
        frozen_layers: Numero di layer iniziali da congelare.

    Returns:
        Modello di classificazione con solo una parte del backbone addestrabile.
    """
    model = load_sequence_classifier(model_id=model_id, num_labels=num_labels)

    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False

    for layer in model.distilbert.transformer.layer[:frozen_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    return model


def count_parameters(model: Any) -> dict[str, float]:
    """
    Conta parametri totali e addestrabili.

    Args:
        model: Modello PyTorch.

    Returns:
        Dizionario con `total`, `trainable` e `trainable_percent`.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "trainable_percent": float(trainable / total * 100 if total else 0.0),
    }
