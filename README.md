# Deep Learning Applications — Portfolio dei laboratori

**Corso:** Deep Learning Applications<br>
**Autore:** Francesco Faggioli

Questa repository contiene i tre laboratori completati per il corso. Il portfolio si sviluppa dal transfer learning per la computer vision, passando per l'adattamento dei Transformer e dei modelli visione-linguaggio, fino al reinforcement learning con metodi policy-gradient. I laboratori sono studi distinti: **solo DLA 3 riguarda il Deep Reinforcement Learning** e le rispettive metriche non devono essere confrontate come se descrivessero un unico compito.

La repository finale è progettata per essere consultata direttamente su GitHub. Gli output eseguiti dei notebook restano visibili, le metriche principali sono replicate in file leggeri nella cartella `results/`, le operazioni costose sono disabilitate per impostazione predefinita e ogni relazione distingue le evidenze misurate dalle interpretazioni. I fondamenti matematici sono documentati nelle relazioni dei laboratori e accanto alle corrispondenti implementazioni nei notebook.

## Panoramica del portfolio

| Laboratorio | Argomento | Metodi principali | Notebook principale della consegna | Relazione dettagliata |
| --- | --- | --- | --- | --- |
| DLA 1 | Riconoscimento dei segnali stradali GTSRB | Feature ResNet preaddestrate, SVM, fine-tuning selettivo, retrieval con similarità coseno, NMC | [`DLA_1.ipynb`](DLA_1/DLA_1.ipynb) | [`DLA_1/README.md`](DLA_1/README.md) |
| DLA 2 | Transformer e adattamento visione-linguaggio | DistilBERT, Hugging Face Trainer, LoRA, congelamento parziale, CLIP-Adapter | [`DLA_2.ipynb`](DLA_2/DLA_2.ipynb) | [`DLA_2/README.md`](DLA_2/README.md) |
| DLA 3 | Deep Reinforcement Learning | REINFORCE, baseline di valore appresa, A2C vettorizzato, selezione della temperatura della policy | [`DLA_3.ipynb`](DLA_3/DLA_3.ipynb) | [`DLA_3/README.md`](DLA_3/README.md) |

I testi ufficiali degli esercizi sono disponibili in [`DLA_1/ASSIGNMENT.md`](DLA_1/ASSIGNMENT.md), [`DLA_2/ASSIGNMENT.md`](DLA_2/ASSIGNMENT.md) e [`DLA_3/ASSIGNMENT.md`](DLA_3/ASSIGNMENT.md).

## Risultati principali

| Laboratorio | Esperimento | Metrica | Risultato | Evidenza versionata |
| --- | --- | ---: | ---: | --- |
| DLA 1 | Feature ResNet-18 + SVM | Accuratezza sul test GTSRB | 0.6412 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv) |
| DLA 1 | Baseline con fine-tuning della sola testa | Accuratezza sul test GTSRB | 0.5038 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv) |
| DLA 1 | Fine-tuning selettivo migliorato | Accuratezza sul test GTSRB | 0.8025 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv) |
| DLA 1 | Retrieval con coseno / NMC | Precision@1 / accuratezza | 0.4812 / 0.4185 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv) |
| DLA 2 | Fine-tuning completo di DistilBERT | Accuratezza sul test Rotten Tomatoes | 0.8443 | [`sentiment_results.csv`](DLA_2/results/sentiment_results.csv) |
| DLA 2 | LoRA | Accuratezza sul test / quota addestrabile | 0.8386 / 1.09% | [`sentiment_results.csv`](DLA_2/results/sentiment_results.csv) |
| DLA 2 | Congelamento parziale | Accuratezza sul test | 0.8377 | [`sentiment_results.csv`](DLA_2/results/sentiment_results.csv) |
| DLA 2 | CLIP-Adapter, bottleneck 128 | Accuratezza su ImageNet-Sketch | 0.5241 | [`clip_results.csv`](DLA_2/results/clip_results.csv) |
| DLA 3 | REINFORCE / baseline di valore | Miglior ritorno periodico su CartPole | 489.35 / 500.00 | [`method_summary.csv`](DLA_3/results/method_summary.csv) |
| DLA 3 | A2C CartPole | Ritorno medio greedy | 494.51 | [`method_summary.csv`](DLA_3/results/method_summary.csv) |
| DLA 3 | A2C LunarLander, campionamento a T=0.75 | Ritorno medio / tasso di successo | 165.76 / 56.0% | [`lunarlander_final_evaluation.json`](DLA_3/results/lunarlander_final_evaluation.json) |

Questi valori riassumono esecuzioni precedenti e non sono stati ricalcolati durante la revisione della documentazione. Il README di ogni laboratorio ne discute la provenienza, la variabilità, i risultati negativi e i limiti.

## Panoramica visuale

### DLA 1 — classificazione GTSRB

![Confronto dell'accuratezza sul test GTSRB](DLA_1/figures/gtsrb_test_accuracy_comparison.svg)

Il fine-tuning selettivo ha prodotto il miglior risultato finale sul test. La baseline neurale con la sola testa addestrabile è rimasta al di sotto della SVM applicata alle feature fisse, motivando l'adattamento selettivo dell'ultimo blocco residuo.

### DLA 2 — adattamento per l'analisi del sentiment

![Accuratezza sul test Rotten Tomatoes](DLA_2/figures/sentiment_test_accuracy.svg)

Il fine-tuning completo ha ottenuto l'accuratezza più alta, mentre LoRA ne ha conservato gran parte ottimizzando circa l'1.09% dei parametri del modello.

### DLA 3 — valutazione su LunarLander

![Analisi della temperatura della policy su LunarLander](DLA_3/figures/lunarlander_temperature_sweep.svg)

Le deviazioni standard elevate giustificano una valutazione ripetuta, anziché la selezione del checkpoint sulla base di un singolo rollout. La policy finale usa azioni campionate con temperatura `0.75`.

## Struttura della repository

```text
DLA_Lab/
├── README.md
├── AI_USAGE.md
├── CODE_OF_CONDUCT.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── tools/
│   ├── audit_submission.py
│   ├── build_report_assets.py
│   └── smoke_notebooks.py
├── DLA_1/
│   ├── DLA_1.ipynb
│   ├── ASSIGNMENT.md
│   ├── README.md
│   └── config/  figures/  results/  notebooks/  scripts/  src/
├── DLA_2/
│   ├── DLA_2.ipynb
│   ├── ASSIGNMENT.md
│   ├── README.md
│   └── config/  figures/  results/  notebooks/  scripts/  src/
└── DLA_3/
    ├── DLA_3.ipynb
    ├── ASSIGNMENT.md
    ├── README.md
    └── config/  figures/  results/  notebooks/  scripts/  src/
```

I dataset, i checkpoint dei modelli, le esecuzioni W&B, le directory di output di Hugging Face, le cache e gli artefatti locali completi sono deliberatamente esclusi da Git.

## Ordine di esecuzione consigliato

1. Leggere questa panoramica del portfolio e il README dettagliato di ogni laboratorio.
2. Aprire i tre notebook indice: [`DLA_1.ipynb`](DLA_1/DLA_1.ipynb), [`DLA_2.ipynb`](DLA_2/DLA_2.ipynb) e [`DLA_3.ipynb`](DLA_3/DLA_3.ipynb).
3. Seguire i notebook tecnici nell'ordine indicato nel rispettivo `notebooks/README.md`.
4. Consultare gli output conservati e le corrispondenti evidenze CSV/JSON nella cartella `results/`.

## Installazione

I tre file di configurazione dell'ambiente hanno ruoli distinti e coerenti:

- `requirements.txt` è l'elenco canonico delle dipendenze di runtime e include `-e .`, così i tre pacchetti locali vengono installati in modalità modificabile;
- `environment.yml` crea l'ambiente Conda e installa tramite pip i requisiti canonici;
- `pyproject.toml` definisce il progetto locale installabile, l'individuazione dei pacchetti, i metadati del progetto e la configurazione di Black/Ruff. Non definisce un elenco concorrente di dipendenze di runtime.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

### Linux o WSL

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m jupyter lab
```

In alternativa:

```bash
conda env create -f environment.yml
conda activate DLA2026
python -m jupyter lab
```

Per DLA 3 è consigliato Linux/WSL, perché le dipendenze di Box2D e del rendering di LunarLander sono più affidabili in quell'ambiente. CUDA è facoltativo; l'esecuzione su CPU è supportata, anche se l'estrazione completa delle feature e l'addestramento risultano sensibilmente più lenti.

## Modalità di consultazione rapida

La configurazione inclusa nel commit è pensata per una consultazione sicura:

- vengono caricati le metriche CSV/JSON e i grafici salvati;
- gli output conservati nei notebook restano visibili su GitHub;
- i flag per addestramento, valutazione completa, estrazione delle feature, W&B, rendering, esportazione video e sovrascrittura dei checkpoint restano disabilitati;
- l'apertura o l'esecuzione dei notebook non installa automaticamente alcun pacchetto.

Lo smoke test non distruttivo esegue copie dei notebook pubblici:

```bash
python tools/smoke_notebooks.py --kernel drl --output /tmp/dla_lab_smoke
```

## Modalità di esperimento completo

La riproduzione completa richiede i dataset non versionati, il download dei modelli, i checkpoint locali ove applicabile, risorse di calcolo adeguate e l'attivazione esplicita dei relativi flag `RUN_*`. L'abilitazione di un flag di addestramento o sovrascrittura costituisce una richiesta intenzionale di ricalcolare un esperimento; non è necessaria per consultare le evidenze presentate.

## Politica di conservazione degli output

Gli output eseguiti fanno parte della consegna. La repository conserva tabelle scientifiche, metriche, curve, report di classificazione, risultati di retrieval, valutazioni di reinforcement learning e riepiloghi visuali leggeri degli episodi. I pesanti payload delle animazioni incorporate sono esclusi, mentre il codice locale per il rendering resta disponibile dietro flag disabilitati.

## Politica per dataset e checkpoint

- I download dei dataset non sono versionati.
- I checkpoint (`*.pt`, `*.pth`, `*.safetensors` e le cartelle di checkpoint di Hugging Face) restano locali a causa delle loro dimensioni.
- Le metriche finali leggere, le cronologie, i dati di selezione e gli input per i grafici sono versionati nella cartella `results/`.
- [`tools/build_report_assets.py`](tools/build_report_assets.py) rigenera i grafici delle relazioni a partire dalle evidenze versionate, senza ripetere l'addestramento.

## Riproducibilità

- I seed globali sono `42` per DLA 1 e DLA 2 e `2112` per DLA 3; i seed degli ambienti e delle valutazioni sono derivati in modo deterministico nell'implementazione.
- Le relazioni riportano le dimensioni esatte delle suddivisioni dei dataset osservate nei notebook eseguiti.
- I risultati stocastici di reinforcement learning sono valutati su più episodi e ne viene riportata la dispersione.
- Non si garantisce la riproducibilità bit per bit tra versioni differenti di CUDA, driver, PyTorch, Gymnasium o sistema operativo.

## Uso dell'IA e integrità accademica

ChatGPT e OpenAI Codex hanno supportato il chiarimento concettuale, il debug, l'organizzazione del codice, la documentazione, la presentazione matematica e i controlli di coerenza. Ogni formula è stata verificata rispetto all'implementazione e gli output dell'IA non sono stati accettati come evidenza sperimentale. La dichiarazione completa è disponibile in [`AI_USAGE.md`](AI_USAGE.md).

## Codice di condotta

Le norme di condotta, l'integrità accademica, l'attribuzione della paternità e la responsabilità sono descritte in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Riferimenti

- [Documentazione di PyTorch](https://pytorch.org/docs/stable/index.html)
- [Documentazione di Torchvision](https://pytorch.org/vision/stable/index.html)
- [Documentazione di Hugging Face](https://huggingface.co/docs)
- [Documentazione di Gymnasium](https://gymnasium.farama.org/)
- [Documentazione di Weights & Biases](https://docs.wandb.ai/)

I dataset, gli articoli, le API, i dettagli matematici e i riferimenti implementativi specifici di ogni laboratorio sono elencati nelle rispettive relazioni e nei notebook.
