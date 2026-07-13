from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


def notebook(cells: list[dict], kernel_display: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": kernel_display,
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SETUP = r"""
from pathlib import Path
import sys

cwd = Path.cwd().resolve()
PROJECT_ROOT = cwd.parent if cwd.name == "notebooks" else cwd
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

PROJECT_ROOT
"""


def write_notebook(name: str, cells: list[dict], kernel: str) -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOK_DIR / name
    path.write_text(json.dumps(notebook(cells, kernel), indent=2), encoding="utf-8")


def build_01() -> list[dict]:
    return [
        md("""
# Esercizio 1 - Analisi del sentiment: dataset, tokenizer e baseline stabile

Kernel consigliato: `DLA2026-transformers`.

Obiettivo: costruire una baseline semplice ma solida per sentiment analysis su Rotten Tomatoes.

Il percorso e' diviso in tre passi:

1. caricare ed esplorare gli split del dataset;
2. caricare tokenizer e modello DistilBERT e osservare gli output;
3. usare DistilBERT come feature extractor e addestrare una SVM lineare.

Disclosure AI: la struttura del notebook e' stata riorganizzata con supporto di ChatGPT/Codex. Il codice e' stato separato in funzioni riusabili e deve essere rieseguito e verificato manualmente prima della consegna.
"""),
        md("""
## Setup

Le funzioni sono nel modulo `src/dla_lab2`. Il notebook resta corto e documenta solo i passaggi principali.

Nota sull'ambiente: questo notebook usa il kernel `DLA2026-transformers`. In questo ambiente puo' comparire un errore interno di `datasets/dill` durante `load_dataset`. La funzione `load_rotten_tomatoes()` gestisce quel caso caricando gli stessi split dalla cache Hugging Face locale gia' presente sul computer.
"""),
        code(SETUP),
        code("""
from dataclasses import asdict
import importlib

import pandas as pd
import torch

import dla_lab2.sentiment as sentiment
from dla_lab2.seed import set_seed

# Ricarichiamo il modulo per evitare che Jupyter tenga in memoria
# una vecchia versione delle funzioni dopo modifiche ai file .py.
importlib.reload(sentiment)

from dla_lab2.sentiment import (
    dataset_overview,
    evaluate_sklearn_classifier,
    extract_cls_features_with_pipeline,
    inspect_tokenizer_output,
    load_distilbert_base,
    load_rotten_tomatoes,
    sample_examples,
    train_linear_svm,
)

# Fissiamo il seed per rendere ripetibili gli esempi casuali e la SVM.
set_seed(42)

# Se e' disponibile CUDA usiamo la GPU per DistilBERT.
# La pipeline Hugging Face usa 0 per la prima GPU e -1 per la CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline_device = 0 if torch.cuda.is_available() else -1
device
"""),
        md("""
## Esercizio 1.1 - Caricamento degli split del dataset

Richiesta dell'esercizio:

- caricare il dataset Cornell Rotten Tomatoes;
- capire quali split sono disponibili;
- esplorare come sono organizzati esempi, colonne e label.

Carichiamo il dataset con Hugging Face Datasets. Se l'ambiente non riesce a usare `load_dataset` per il problema `Pickler._batch_setitems`, la funzione usa il fallback sulla cache locale.
"""),
        code("""
# Carichiamo tutti gli split disponibili.
ds_dict = load_rotten_tomatoes()

# Questa tabella mostra quanti esempi ci sono in ogni split,
# quali colonne sono disponibili e quali label sono presenti.
pd.DataFrame(dataset_overview(ds_dict))
"""),
        code("""
# Verifichiamo come si accede agli split e a un singolo esempio.
print("Split disponibili:", list(ds_dict.keys()))
print("Primo esempio train:")
ds_dict["train"][0]
"""),
        code("""
# Controlliamo il bilanciamento delle classi in ogni split.
# La label 0 indica sentiment negativo, la label 1 indica sentiment positivo.
label_counts = {
    split: pd.Series(ds_dict[split]["label"]).value_counts().sort_index()
    for split in ds_dict.keys()
}
pd.DataFrame(label_counts).T.rename(columns={0: "negative", 1: "positive"})
"""),
        code("""
# Guardiamo alcuni esempi casuali per capire il tipo di frasi.
pd.DataFrame(sample_examples(ds_dict["train"], n=8, seed=42))
"""),
        md("""
Osservazioni sull'Esercizio 1.1:

- il dataset e' gia' diviso in `train`, `validation` e `test`;
- ogni esempio contiene almeno `text` e `label`;
- le label sono binarie: `0` negativo, `1` positivo;
- gli split sono gia' pronti per training e valutazione, quindi non creiamo un nuovo split manuale.
"""),
        md("""
## Esercizio 1.2 - BERT pre-addestrato e tokenizer

Richiesta dell'esercizio:

- caricare tokenizer e modello DistilBERT;
- tokenizzare alcuni esempi;
- passare i token nel modello;
- capire quali output vengono prodotti.

Usiamo `AutoTokenizer` e `AutoModel` dentro la funzione `load_distilbert_base()`. Il modello caricato non ha testa di classificazione: restituisce rappresentazioni contestuali dei token.
"""),
        code("""
# Carichiamo tokenizer e modello base.
tokenizer, bert_model = load_distilbert_base(device=device)

# Selezioniamo tre frasi dal train split e le tokenizziamo.
texts = [ds_dict["train"][i]["text"] for i in range(3)]
batch = inspect_tokenizer_output(tokenizer, texts)

# Il tokenizer restituisce tensori come input_ids e attention_mask.
batch.keys()
"""),
        code("""
# Decodifichiamo la prima sequenza per vedere token speciali e padding.
print(tokenizer.decode(batch["input_ids"][0]))

# Controlliamo la forma dei tensori in input al modello.
print({key: value.shape for key, value in batch.items()})
"""),
        code("""
# Passiamo il batch nel modello senza calcolare gradienti:
# qui stiamo solo ispezionando gli output, non addestrando.
with torch.no_grad():
    outputs = bert_model(**{key: value.to(device) for key, value in batch.items()})

print(outputs.keys())
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
"""),
        code("""
# Il primo token e' [CLS]. Il suo vettore nell'ultimo layer
# viene spesso usato come rappresentazione sintetica della frase.
cls_vectors = outputs.last_hidden_state[:, 0, :]
cls_vectors.shape
"""),
        md("""
Osservazioni sull'Esercizio 1.2:

- `input_ids` contiene gli identificativi numerici dei token;
- `attention_mask` distingue token reali e padding;
- `last_hidden_state` ha forma `(batch, sequence_length, hidden_size)`;
- per DistilBERT base `hidden_size` e' 768;
- per la baseline useremo `last_hidden_state[:, 0, :]`, cioe' il vettore del token `[CLS]` dell'ultimo layer.
"""),
        md("""
## Esercizio 1.3 - Baseline stabile

Richiesta dell'esercizio:

1. usare DistilBERT come feature extractor;
2. addestrare un classificatore classico sulle feature estratte;
3. valutare validation e test.

Qui usiamo la pipeline Hugging Face `feature-extraction`. La funzione `extract_cls_features_with_pipeline()` interpreta l'output della pipeline e conserva solo il token `[CLS]` dell'ultimo layer. Poi addestriamo una SVM lineare di Scikit-learn.
"""),
        code("""
features = {}
for split in ["train", "validation", "test"]:
    # Convertiamo la colonna testuale in lista per passarla alla pipeline.
    texts = list(ds_dict[split]["text"])

    # Estraiamo una matrice: una riga per frase, 768 colonne per le feature CLS.
    features[split] = extract_cls_features_with_pipeline(
        texts=texts,
        model=bert_model,
        tokenizer=tokenizer,
        batch_size=32,
        device=pipeline_device,
    )

    # Controllo atteso: train -> (8530, 768), validation/test -> (1066, 768).
    print(split, features[split].shape)
"""),
        code("""
# Addestriamo la SVM solo sul train split.
svm = train_linear_svm(features["train"], ds_dict["train"]["label"])

# Valutiamo sugli split non usati per addestrare il classificatore.
baseline_metrics = [
    asdict(evaluate_sklearn_classifier(svm, features["validation"], ds_dict["validation"]["label"], "validation")),
    asdict(evaluate_sklearn_classifier(svm, features["test"], ds_dict["test"]["label"], "test")),
]
baseline_table = pd.DataFrame(baseline_metrics)
baseline_table
"""),
        md("""
Interpretazione della tabella:

- `accuracy`: percentuale di frasi classificate correttamente;
- `f1`: equilibrio tra precision e recall sulla classe positiva;
- `precision`: quando il modello predice positivo, quanto spesso ha ragione;
- `recall`: quanti positivi reali riesce a recuperare.

Questi numeri sono la baseline stabile da confrontare con il fine-tuning del notebook successivo.
"""),
        md("""
## Conclusioni dell'Esercizio 1

- L'Esercizio 1.1 e' soddisfatto perche' abbiamo caricato gli split, controllato colonne, dimensioni, label ed esempi.
- L'Esercizio 1.2 e' soddisfatto perche' abbiamo caricato DistilBERT e tokenizer, tokenizzato frasi reali e ispezionato gli output del modello.
- L'Esercizio 1.3 e' soddisfatto perche' abbiamo estratto feature `[CLS]`, addestrato una SVM e valutato validation e test.

Commento finale: questa baseline non modifica DistilBERT. Per questo e' piu' stabile e meno costosa del fine-tuning completo, ma puo' essere meno accurata perche' il transformer non si adatta al dataset Rotten Tomatoes.
"""),
    ]


def build_02() -> list[dict]:
    return [
        md("""
# Esercizio 2 - Fine-tuning completo di DistilBERT

Kernel usato: `DLA2026-transformers`.

Obiettivo: passare dalla baseline stabile del notebook 1 al fine-tuning supervisionato di DistilBERT con una testa di classificazione per sentiment analysis.

In questo notebook verifichiamo:

1. tokenizzazione degli split con `Dataset.map`;
2. modello `AutoModelForSequenceClassification` con classification head;
3. setup completo di `Trainer`, `TrainingArguments`, `DataCollatorWithPadding` e metriche Scikit-learn;
4. training per 3 epoche e valutazione finale su validation e test.
"""),
        code(SETUP),
        code("""
from dataclasses import asdict
import importlib
import math

import pandas as pd
import torch

import dla_lab2.sentiment as sentiment
from dla_lab2.paths import output_dir
from dla_lab2.seed import set_seed

# Ricarichiamo il modulo per usare l'ultima versione degli helper Trainer.
importlib.reload(sentiment)

from dla_lab2.sentiment import (
    build_trainer,
    build_training_arguments,
    disable_notebook_progress_callback,
    load_rotten_tomatoes,
    load_sequence_classifier,
    load_distilbert_base,
    tokenize_dataset_splits,
)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
device
"""),
        md("""
## Esercizio 2.1 - Pre-elaborazione dei token

Richiesta dell'esercizio: tokenizzare gli split una sola volta con Hugging Face `Dataset.map`, evitando di ritokenizzare a ogni batch.

Il padding non viene fatto nella tokenizzazione: lo fara' `DataCollatorWithPadding` durante la costruzione dei batch. Questo e' piu' efficiente perche' ogni batch viene paddato solo alla lunghezza massima presente nel batch stesso.
"""),
        code("""
# Carichiamo dataset e tokenizer.
ds_dict = load_rotten_tomatoes()
tokenizer, _ = load_distilbert_base(device=None)

# Dataset.map applica il tokenizer a train, validation e test.
tokenized = tokenize_dataset_splits(ds_dict, tokenizer, max_length=256)

# Verifica richiesta: ogni elemento contiene text, label, input_ids e attention_mask.
tokenized["train"][0].keys()
"""),
        code("""
# Ispezioniamo un esempio tokenizzato.
# La run mostra che text e label sono rimasti disponibili,
# mentre input_ids e attention_mask sono tensori PyTorch.
example = tokenized["train"][0]
print(example["text"])
print(example["label"])
print(example["input_ids"].shape)
print(example["attention_mask"].shape)
"""),
        md("""
Osservazioni sull'Esercizio 2.1:

- `Dataset.map` e' stato eseguito su tutti e tre gli split.
- Ogni esempio mantiene `text` e `label`.
- Ogni esempio aggiunge `input_ids` e `attention_mask`.
- Nella run, il primo esempio ha 47 token prima del padding dinamico di batch.
- La colonna `token_type_ids` compare per compatibilita' del tokenizer, ma per DistilBERT non e' centrale in questo esercizio.
"""),
        md("""
## Esercizio 2.2 - Modello per la classificazione di sequenze

Richiesta dell'esercizio: preparare DistilBERT per un task di sequence classification.

Usiamo `AutoModelForSequenceClassification`, che carica il backbone DistilBERT pre-addestrato e aggiunge una testa di classificazione binaria. I pesi della testa (`pre_classifier` e `classifier`) sono nuovi e quindi devono essere addestrati sul dataset Rotten Tomatoes.
"""),
        code("""
# Istanza del modello con testa di classificazione binaria.
# I pesi MISSING nel report sono attesi: corrispondono alla nuova testa di classificazione.
classifier_model = load_sequence_classifier(num_labels=2)
classifier_model
"""),
        md("""
## Esercizio 2.3 - Fine-tuning con Trainer

Richieste dell'esercizio:

1. usare `DataCollatorWithPadding` per creare batch con padding dinamico;
2. definire metriche di valutazione da logits e labels;
3. impostare `TrainingArguments` ragionevoli;
4. creare un `Trainer` con train/validation split;
5. chiamare `trainer.train()` e poi `trainer.evaluate()`.

Queste parti sono implementate negli helper:

- `build_trainer()` crea `DataCollatorWithPadding`, collega `compute_sklearn_metrics` e istanzia `Trainer`;
- `compute_sklearn_metrics` calcola accuracy, F1, precision e recall con Scikit-learn;
- `build_training_arguments()` imposta learning rate, batch size, epoche, checkpoint per epoca, scheduler cosine e best checkpoint su accuracy.

Nota tecnica: rimuoviamo il `NotebookProgressCallback` per evitare l'errore `on_train_begin must be called before on_evaluate`. Per rendere comunque chiara la run, aggiungiamo sotto una tabella esplicita con le metriche per epoca.
"""),
        code("""
def model_init():
    # Ogni run parte da una nuova istanza pulita del modello pre-addestrato.
    return load_sequence_classifier(num_labels=2)


num_epochs = 3
batch_size = 32
total_steps = math.ceil(len(tokenized["train"]) / batch_size) * num_epochs
warmup_steps = math.ceil(total_steps * 0.1)

training_args = build_training_arguments(
    output_dir=output_dir("distilbert_full_finetuning"),
    learning_rate=2e-5,
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    warmup_steps=warmup_steps,
)

trainer = build_trainer(
    tokenizer=tokenizer,
    training_args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    model_init=model_init,
)

print(f"Total training steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")
"""),
        code("""
# Avvio del fine-tuning completo di DistilBERT.
# Il callback grafico del notebook e' disattivato, quindi il riepilogo leggibile
# delle epoche viene costruito nella cella successiva da trainer.state.log_history.
train_output = trainer.train()
train_output
"""),
        code("""
# Riepilogo leggibile del training.
# `loss_log` mostra la training loss registrata ogni 50 step.
# `epoch_eval_log` mostra le metriche validation alla fine di ogni epoca.
history = pd.DataFrame(trainer.state.log_history)

loss_log = history[history["loss"].notna()][["epoch", "step", "loss", "learning_rate", "grad_norm"]].copy()
epoch_eval_log = history[history["eval_loss"].notna()][
    ["epoch", "step", "eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall"]
].copy()

display(loss_log)
display(epoch_eval_log)
"""),
        code("""
# Valutiamo esplicitamente il best model su validation e test.
# La funzione sotto rimuove il callback notebook anche se il Trainer era stato creato
# in una sessione precedente.
disable_notebook_progress_callback(trainer)
validation_metrics = trainer.evaluate(tokenized["validation"])
test_metrics = trainer.evaluate(tokenized["test"])

pd.DataFrame([validation_metrics, test_metrics], index=["validation", "test"])
"""),
        md("""
## Conclusioni dell'Esercizio 2

Tutti i punti richiesti sono stati svolti.

Esercizio 2.1:
- gli split sono stati tokenizzati con `Dataset.map`;
- ogni elemento contiene `text`, `label`, `input_ids` e `attention_mask`;
- il padding e' lasciato al data collator, quindi avviene dinamicamente per batch.

Esercizio 2.2:
- DistilBERT e' stato istanziato come `AutoModelForSequenceClassification`;
- la testa di classificazione binaria e' nuova e viene addestrata sul task;
- i messaggi `MISSING` per `pre_classifier` e `classifier` sono attesi.

Esercizio 2.3:
- `DataCollatorWithPadding` e' usato dentro `build_trainer()`;
- le metriche sono calcolate con Scikit-learn da logits e labels;
- `TrainingArguments` usa 3 epoche, batch size 32, learning rate `2e-5`, weight decay `0.01`, scheduler cosine e warmup di 81 step;
- `Trainer` e' stato addestrato per 3 epoche;
- validation e test sono stati valutati esplicitamente.

Risultati osservati:

- validation accuracy: `0.8527`, F1: `0.8517`;
- test accuracy: `0.8443`, F1: `0.8428`;
- train loss media finale riportata da `TrainOutput`: `0.3052`;
- training runtime: circa `236 s`;
- best checkpoint scelto per accuracy: epoca 3.

Confronto con la baseline del notebook 1:

- baseline SVM test accuracy: `0.7946`;
- fine-tuning DistilBERT test accuracy: `0.8443`;
- miglioramento assoluto: circa `+0.0497` punti di accuracy.

Interpretazione: il fine-tuning completo migliora chiaramente la baseline congelata. La validation loss e' minima all'epoca 2 ma l'accuracy migliore arriva all'epoca 3; siccome il criterio scelto e' accuracy, il checkpoint finale e' coerente con la configurazione. Il risultato e' adatto come riferimento principale per gli esercizi successivi su fine-tuning efficiente.
"""),
    ]


def build_03() -> list[dict]:
    return [
        md("""
# Esercizio 3.1 - Fine-tuning efficiente per l'analisi del sentiment

Kernel consigliato: `DLA2026-transformers`.

Obiettivo: ridurre i parametri addestrabili rispetto al fine-tuning completo. Confrontiamo due strategie: LoRA e partial freezing.
"""),
        code(SETUP),
        code("""
import pandas as pd
import importlib
import math
import torch

import dla_lab2.sentiment as sentiment
from dla_lab2.paths import output_dir
from dla_lab2.seed import set_seed

importlib.reload(sentiment)

from dla_lab2.sentiment import (
    build_trainer,
    build_training_arguments,
    count_parameters,
    disable_notebook_progress_callback,
    load_distilbert_base,
    load_rotten_tomatoes,
    lora_sequence_classifier_init,
    partial_freezing_sequence_classifier_init,
    suppress_subprocess_reader_unicode_errors,
    tokenize_dataset_splits,
)

set_seed(42)
suppress_subprocess_reader_unicode_errors()
device = "cuda" if torch.cuda.is_available() else "cpu"
device
"""),
        code("""
ds_dict = load_rotten_tomatoes()
tokenizer, _ = load_distilbert_base(device=None)
tokenized = tokenize_dataset_splits(ds_dict, tokenizer, max_length=256)

num_epochs = 3
batch_size = 32
total_steps = math.ceil(len(tokenized["train"]) / batch_size) * num_epochs
warmup_steps = math.ceil(total_steps * 0.1)
"""),
        md("""
## Strategia A - LoRA

LoRA aggiunge matrici a basso rango nei layer di attenzione. Qui modifichiamo solo `q_lin` e `v_lin`, lasciando il resto del modello congelato o quasi.

Usiamo `warmup_steps` esplicito e sopprimiamo un traceback Unicode non fatale che puo' comparire in questo ambiente durante controlli subprocess di librerie PEFT.
"""),
        code("""
def lora_model_init():
    return lora_sequence_classifier_init(rank=8, alpha=16, dropout=0.1)


lora_args = build_training_arguments(
    output_dir=output_dir("distilbert_lora"),
    learning_rate=1e-3,
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    warmup_steps=warmup_steps,
)

lora_trainer = build_trainer(
    tokenizer=tokenizer,
    training_args=lora_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    model_init=lora_model_init,
)
"""),
        code("""
lora_train_output = lora_trainer.train()
disable_notebook_progress_callback(lora_trainer)
lora_test = lora_trainer.evaluate(tokenized["test"])
lora_params = count_parameters(lora_trainer.model)
lora_test, lora_params
"""),
        md("""
## Strategia B - Partial Freezing

Congeliamo embeddings e i primi 4 layer transformer. Addestriamo solo gli ultimi layer e la testa. E' meno specializzato di LoRA, ma semplice da spiegare.
"""),
        code("""
def freeze_model_init():
    return partial_freezing_sequence_classifier_init(frozen_layers=4)


freeze_args = build_training_arguments(
    output_dir=output_dir("distilbert_partial_freeze"),
    learning_rate=2e-5,
    train_batch_size=batch_size,
    eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    warmup_steps=warmup_steps,
)

freeze_trainer = build_trainer(
    tokenizer=tokenizer,
    training_args=freeze_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    model_init=freeze_model_init,
)
"""),
        code("""
freeze_train_output = freeze_trainer.train()
disable_notebook_progress_callback(freeze_trainer)
freeze_test = freeze_trainer.evaluate(tokenized["test"])
freeze_params = count_parameters(freeze_trainer.model)
freeze_test, freeze_params
"""),
        code("""
comparison = pd.DataFrame(
    [
        {
            "method": "Fine-tuning completo (Esercizio 2)",
            "test_accuracy": 0.844278,
            "test_f1": 0.842803,
            "trainable_percent": 100.0,
        },
        {
            "method": "LoRA",
            "test_accuracy": lora_test.get("eval_accuracy"),
            "test_f1": lora_test.get("eval_f1"),
            "trainable_percent": lora_params["trainable_percent"],
        },
        {
            "method": "Partial freezing",
            "test_accuracy": freeze_test.get("eval_accuracy"),
            "test_f1": freeze_test.get("eval_f1"),
            "trainable_percent": freeze_params["trainable_percent"],
        },
    ]
)
comparison
"""),
        md("""
## Conclusioni dell'Esercizio 3.1

- Il confronto corretto non e' solo accuracy: bisogna guardare anche la percentuale di parametri addestrabili.
- LoRA di solito e' la scelta piu' pulita per PEFT perche' mantiene quasi tutto il modello originale congelato.
- Partial freezing e' utile come baseline efficiente, ma puo' perdere capacita' se si congelano troppi layer.
"""),
    ]


def build_04() -> list[dict]:
    return [
        md("""
# Esercizio 3.2 - Adattamento efficiente di CLIP

Kernel consigliato: `clip_lora`.

Obiettivo: valutare CLIP zero-shot su ImageNet-Sketch e poi adattarlo con un metodo parameter-efficient. Usiamo CLIP-Adapter sulle feature immagine perche' e' leggero, chiaro e adatto a GPU con poca VRAM.

Nota: questo notebook usa un adapter, non una LoRA interna al transformer. E' comunque parameter-efficient: il backbone CLIP resta congelato e si addestra solo un piccolo MLP.
"""),
        code(SETUP),
        code("""
from torch.utils.data import DataLoader
import pandas as pd
import torch

from dla_lab2.clip_utils import (
    CLIPAdapter,
    build_clip_tensor_dataset,
    build_text_features,
    build_text_features_ensemble,
    evaluate_adapter,
    evaluate_precomputed_features,
    load_imagenet_labels,
    load_imagenet_sketch,
    load_open_clip_model,
    precompute_image_features,
    split_tensor_dataset,
    train_clip_adapter,
)
from dla_lab2.seed import set_seed

set_seed(42)
"""),
        md("""
## 1. Modello e labels

Carichiamo CLIP ViT-B/16 e i nomi semplici delle 1000 classi ImageNet. Usiamo la variante quickgelu per coerenza con i pesi OpenAI.
"""),
        code("""
model, preprocess, tokenizer, device = load_open_clip_model(
    model_name="ViT-B-16-quickgelu",
    pretrained="openai",
)

imagenet_class_names = load_imagenet_labels(PROJECT_ROOT / "imagenet_labels.json")
print(device)
print(len(imagenet_class_names), imagenet_class_names[:5])
"""),
        md("""
## 2. Dataset ImageNet-Sketch

ImageNet-Sketch e' piu' interessante di ImageNette perche' introduce domain shift: CLIP vede disegni e schizzi invece di foto naturali.
"""),
        code("""
sketch_train, sketch_val = load_imagenet_sketch(seed=42, train_fraction=0.8)
print(len(sketch_train), len(sketch_val))
"""),
        code("""
TRAIN_SAMPLES = 5000
BATCH_SIZE = 64

# Usiamo un sottoinsieme del training split per mantenere il laboratorio gestibile.
# La validation esterna resta completa, cosi' la valutazione finale e' piu' stabile.
train_tensor_ds = build_clip_tensor_dataset(sketch_train, preprocess, num_samples=TRAIN_SAMPLES)
val_tensor_ds = build_clip_tensor_dataset(sketch_val, preprocess, num_samples=None)

train_image_loader = DataLoader(train_tensor_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
val_image_loader = DataLoader(val_tensor_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f"Full train split examples: {len(sketch_train)}")
print(f"Train subset examples used: {len(train_tensor_ds)}")
print(f"External validation examples used: {len(val_tensor_ds)}")
print(f"Image batch size: {BATCH_SIZE}")
"""),
        md("""
## 3. Zero-shot baseline e prompt study

Prima valutiamo CLIP senza training e confrontiamo alcuni prompt semplici. CLIP e' sensibile al testo usato come descrizione della classe: cambiare prompt puo' cambiare leggermente l'accuracy.

Il dataset e' stato diviso inizialmente in modo standard `80/20`: `40.711` esempi nello split train e `10.178` nello split validation. Per il training dell'adapter usiamo solo `5.000` esempi del train split, scelta dichiarata e riproducibile. La validation esterna rimane completa per avere una stima piu' affidabile del risultato finale.

Best practice computazionale: il visual encoder di CLIP resta congelato, quindi calcoliamo le feature immagine del validation set una sola volta e poi riusiamo quelle feature per tutti i prompt.
"""),
        code("""
prompt_templates = [
    "a sketch of a {}",
    "a black and white sketch of a {}",
    "a hand-drawn sketch of a {}",
    "a pencil drawing of a {}",
    "a line drawing of a {}",
]

logit_scale = model.logit_scale.exp().item()

val_feature_ds = precompute_image_features(model, val_image_loader, device)
val_feature_loader = DataLoader(val_feature_ds, batch_size=256, shuffle=False)

prompt_results = []
for template in prompt_templates:
    text_features = build_text_features(model, tokenizer, imagenet_class_names, template, device)
    acc = evaluate_precomputed_features(val_feature_loader, text_features, logit_scale, device)
    prompt_results.append({"prompt": template, "accuracy": acc})

prompt_table = pd.DataFrame(prompt_results).sort_values("accuracy", ascending=False)
prompt_table
"""),
        md("""
Osservazioni attese dalla run:

- confrontare piu' prompt serve a scegliere una formulazione testuale ragionevole prima di addestrare l'adapter;
- con `prompt ensemble` intendiamo la media delle feature testuali prodotte da piu' prompt per la stessa classe.
"""),
        md("""
## 4. CLIP-Adapter

Congeliamo CLIP e addestriamo solo un adapter MLP sulle feature immagine gia' precalcolate. Questo riduce memoria e rischio di danneggiare le rappresentazioni zero-shot.

Il `bottleneck` e' la dimensione del livello nascosto dell'adapter. Un bottleneck piu' piccolo usa meno parametri e forza una correzione piu' semplice; un bottleneck piu' grande e' piu' flessibile ma puo' overfittare piu' facilmente.

Parametri addestrabili dell'adapter:

- bottleneck 64: `66.113` parametri;
- bottleneck 128: `131.713` parametri.
"""),
        code("""
train_feature_ds = precompute_image_features(model, train_image_loader, device)
adapter_train_ds, adapter_val_ds = split_tensor_dataset(train_feature_ds, train_fraction=0.9, seed=42)

adapter_train_loader = DataLoader(adapter_train_ds, batch_size=256, shuffle=True)
adapter_val_loader = DataLoader(adapter_val_ds, batch_size=256, shuffle=False)

print(f"Train feature examples: {len(train_feature_ds)}")
print(f"Adapter train/validation split: {len(adapter_train_ds)} / {len(adapter_val_ds)}")
print(f"External validation feature examples: {len(val_feature_ds)}")
"""),
        code("""
base_prompt = prompt_table.iloc[0]["prompt"]
text_features_base = build_text_features(model, tokenizer, imagenet_class_names, base_prompt, device)

adapter64 = CLIPAdapter(feat_dim=512, bottleneck=64, alpha=0.6)
history64 = train_clip_adapter(
    adapter64,
    adapter_train_loader,
    adapter_val_loader,
    text_features_base,
    logit_scale,
    device,
    epochs=30,
    lr=2e-3,
)

adapter128 = CLIPAdapter(feat_dim=512, bottleneck=128, alpha=0.6)
history128 = train_clip_adapter(
    adapter128,
    adapter_train_loader,
    adapter_val_loader,
    text_features_base,
    logit_scale,
    device,
    epochs=30,
    lr=2e-3,
)
"""),
        md("""
## 5. Valutazione finale

Valutiamo zero-shot, adapter singolo e adapter con prompt ensembling sul validation set esterno ImageNet-Sketch. Usiamo sempre le feature immagine precalcolate, quindi il confronto finale non riesegue inutilmente il visual encoder.
"""),
        code("""
zeroshot_acc = evaluate_precomputed_features(val_feature_loader, text_features_base, logit_scale, device)
acc64 = evaluate_adapter(adapter64, val_feature_loader, text_features_base, logit_scale, device)
acc128 = evaluate_adapter(adapter128, val_feature_loader, text_features_base, logit_scale, device)

text_features_ensemble = build_text_features_ensemble(model, tokenizer, imagenet_class_names, prompt_templates, device)
zs_ensemble = evaluate_precomputed_features(val_feature_loader, text_features_ensemble, logit_scale, device)
acc64_ensemble = evaluate_adapter(adapter64, val_feature_loader, text_features_ensemble, logit_scale, device)
acc128_ensemble = evaluate_adapter(adapter128, val_feature_loader, text_features_ensemble, logit_scale, device)

results = pd.DataFrame(
    [
        {"method": "Zero-shot CLIP", "accuracy": zeroshot_acc, "gain": 0.0},
        {"method": "Zero-shot CLIP + prompt ensemble", "accuracy": zs_ensemble, "gain": zs_ensemble - zeroshot_acc},
        {"method": "CLIP-Adapter bottleneck=64", "accuracy": acc64, "gain": acc64 - zeroshot_acc},
        {"method": "CLIP-Adapter b=64 + prompt ensemble", "accuracy": acc64_ensemble, "gain": acc64_ensemble - zeroshot_acc},
        {"method": "CLIP-Adapter bottleneck=128", "accuracy": acc128, "gain": acc128 - zeroshot_acc},
        {"method": "CLIP-Adapter b=128 + prompt ensemble", "accuracy": acc128_ensemble, "gain": acc128_ensemble - zeroshot_acc},
    ]
)
results
"""),
        code("""
# Questa tabella serve solo a controllare l'andamento della loss nelle ultime epoche.
# La scelta finale del modello si basa sulla tabella di accuracy esterna sopra.
loss_tail = pd.concat(
    [
        pd.DataFrame(history64).assign(adapter="bottleneck=64").tail(),
        pd.DataFrame(history128).assign(adapter="bottleneck=128").tail(),
    ],
    ignore_index=True,
)
loss_tail
"""),
        md("""
Osservazioni dalla valutazione finale:

- baseline zero-shot: da confrontare con gli adapter tramite la colonna `gain`;
- prompt ensemble senza training: puo' migliorare leggermente la baseline, ma non e' garantito che migliori anche gli adapter;
- se l'ensemble dei prompt non migliora l'adapter, si mantiene il miglior prompt singolo;
- train loss molto bassa e validation loss piu' alta indicano possibile overfitting, da discutere nelle conclusioni.
"""),
        md("""
## Conclusioni dell'Esercizio 3.2

Tutti i punti richiesti sono stati svolti.

- Abbiamo usato un modello CLIP piccolo e standard: `ViT-B-16-quickgelu` con pesi OpenAI.
- Abbiamo scelto ImageNet-Sketch, un dataset adatto per testare domain shift perche' contiene schizzi invece di foto naturali.
- Abbiamo valutato CLIP in zero-shot con piu' prompt.
- Abbiamo applicato un metodo parameter-efficient: CLIP-Adapter sulle feature immagine, lasciando congelati image encoder e text encoder.
- Il risultato finale va interpretato come gain rispetto allo zero-shot.

Interpretazione: CLIP parte gia' da una baseline forte su ImageNet-Sketch, ma un piccolo adapter puo' migliorare il dominio sketch senza aggiornare tutto il modello. Se la loss suggerisce overfitting, il risultato resta accettabile quando il miglioramento sulla validation esterna e' misurato, confrontato con la baseline e spiegato.

Best practice adottate:

- split iniziale `80/20` del dataset;
- uso dichiarato di un sottoinsieme del train split per rendere il laboratorio gestibile;
- validation esterna completa per il confronto finale;
- backbone CLIP congelato;
- feature immagine precalcolate;
- confronto con zero-shot, prompt singolo e prompt ensemble;
- reporting del gain rispetto alla baseline;
- scelta finale basata su accuracy esterna, non solo su train loss.
"""),
    ]


def main() -> None:
    write_notebook(
        "01_sentiment_dataset_tokenizer_baseline.ipynb", build_01(), "DLA2026-transformers"
    )
    write_notebook("02_distilbert_full_finetuning.ipynb", build_02(), "DLA2026-transformers")
    write_notebook("03_efficient_finetuning_sentiment.ipynb", build_03(), "DLA2026-transformers")
    write_notebook("04_clip_adapter_imagenet_sketch.ipynb", build_04(), "clip_lora")


if __name__ == "__main__":
    main()
