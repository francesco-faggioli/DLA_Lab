# Deep Learning Applications — Portfolio dei laboratori

**Corso:** Deep Learning Applications

**Autore:** Francesco Faggioli

Questa repository raccoglie i tre laboratori svolti per il corso. Il percorso procede dal transfer learning per la visione artificiale all'adattamento dei Transformer e dei modelli multimodali, fino ai metodi policy-gradient. I laboratori sono studi separati: **solo DLA 3 riguarda il Deep Reinforcement Learning** e le rispettive metriche non devono essere confrontate come se appartenessero a un unico compito.

L'obiettivo principale di questa versione finale è la verificabilità. Gli output eseguiti dei notebook restano visibili, le metriche principali sono replicate in file leggeri nelle cartelle `results/` e ogni relazione distingue l'evidenza osservata dall'interpretazione.

## Repository in sintesi

| Laboratorio | Argomento | Esercizi principali | Metodi principali | Notebook finale | Relazione dettagliata |
| --- | --- | --- | --- | --- | --- |
| DLA 1 | Riconoscimento dei segnali stradali GTSRB | EDA, baseline stabile, fine-tuning, pipeline riutilizzabile, retrieval | Feature ResNet pre-addestrate, SVM, fine-tuning selettivo, cosine retrieval, NMC | [`DLA_1.ipynb`](DLA_1/DLA_1.ipynb) | [`DLA_1/README.md`](DLA_1/README.md) |
| DLA 2 | Transformer e adattamento visione-linguaggio | Baseline di sentiment, fine-tuning completo ed efficiente, adattamento CLIP | DistilBERT, SVM lineare, Hugging Face Trainer, LoRA, congelamento parziale, CLIP-Adapter | [`DLA_2.ipynb`](DLA_2/DLA_2.ipynb) | [`DLA_2/README.md`](DLA_2/README.md) |
| DLA 3 | Deep Reinforcement Learning | Valutazione REINFORCE, value baseline, A2C su due ambienti | Monte Carlo policy gradient, baseline appresa, A2C vettorizzato, selezione della temperatura della policy | [`DLA_3.ipynb`](DLA_3/DLA_3.ipynb) | [`DLA_3/README.md`](DLA_3/README.md) |

I testi ufficiali degli esercizi sono disponibili in [`DLA_1/ASSIGNMENT.md`](DLA_1/ASSIGNMENT.md), [`DLA_2/ASSIGNMENT.md`](DLA_2/ASSIGNMENT.md) e [`DLA_3/ASSIGNMENT.md`](DLA_3/ASSIGNMENT.md).

## Flusso di lavoro complessivo

**DLA 1** parte da una rappresentazione pre-addestrata fissa. Le feature di ResNet-18 forniscono una baseline SVM stabile prima di passare a fine-tuning supervisionato, esperimenti guidati da configurazione, sbilanciamento delle classi, data augmentation, tracciamento degli esperimenti e retrieval senza addestramento.

**DLA 2** trasferisce la stessa idea a dati linguistici e multimodali. Gli embedding congelati di DistilBERT definiscono una baseline; il fine-tuning completo viene poi confrontato con LoRA e congelamento parziale. Un esperimento separato su ImageNet-Sketch valuta CLIP in presenza di domain shift e adatta soltanto un piccolo MLP.

**DLA 3** cambia paradigma di apprendimento. Inizia con REINFORCE episodico, aggiunge una value baseline per ridurre la varianza del gradiente e implementa aggiornamenti actor-critic con ambienti vettorizzati. CartPole convalida l'implementazione; LunarLander mette in evidenza l'instabilità e le esigenze di valutazione del problema di controllo più difficile.

## Risultati principali

| Lab | Esperimento | Metrica | Risultato | Evidenza |
| --- | --- | ---: | ---: | --- |
| DLA 1 | Feature ResNet-18 + SVM | Accuracy di test GTSRB | 0.6412 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv), notebook della baseline eseguito |
| DLA 1 | Fine-tuning selettivo migliorato | Accuracy di test GTSRB | 0.8025 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv), notebook dei miglioramenti eseguito |
| DLA 1 | Cosine retrieval | Precision@1 | 0.4812 | [`test_metrics.csv`](DLA_1/results/test_metrics.csv), notebook del retrieval eseguito |
| DLA 2 | Fine-tuning completo di DistilBERT | Accuracy di test Rotten Tomatoes | 0.8443 | [`sentiment_results.csv`](DLA_2/results/sentiment_results.csv) |
| DLA 2 | LoRA | Accuracy di test / quota addestrabile | 0.8386 / 1.09% | [`sentiment_results.csv`](DLA_2/results/sentiment_results.csv) |
| DLA 2 | CLIP-Adapter, bottleneck 128 | Accuracy ImageNet-Sketch | 0.5241 | [`clip_results.csv`](DLA_2/results/clip_results.csv) |
| DLA 3 | REINFORCE con value baseline | Return periodico finale CartPole | 500.0 | [`method_summary.csv`](DLA_3/results/method_summary.csv) |
| DLA 3 | A2C, policy campionata con T=0.75 | Return medio / success rate LunarLander | 165.76 / 56.0% | [`lunarlander_final_evaluation.json`](DLA_3/results/lunarlander_final_evaluation.json) |

Queste righe riassumono output già eseguiti; non costituiscono nuove affermazioni sperimentali calcolate durante la revisione. In DLA 1, il risultato di test `0.8025` è legato alla valutazione finale fissa del notebook, mentre le cronologie di validazione esportate separatamente descrivono le run di sviluppo denominate. Le accuracy di DLA 2 appartengono a compiti e dataset differenti. La relazione dettagliata di DLA 3 riporta sia la tendenza centrale sia la variabilità.

## Panoramica visuale

### DLA 1 — classificazione GTSRB

![Confronto dell'accuracy di test GTSRB](DLA_1/figures/gtsrb_test_accuracy_comparison.svg)

Il fine-tuning selettivo ha prodotto il miglior risultato finale sul test. La baseline head-only ha ottenuto prestazioni inferiori alla SVM su feature fisse, mostrando che sostituire e addestrare soltanto la testa di classificazione non era sufficiente per questo domain shift. Fonte: [`DLA_1/results/test_metrics.csv`](DLA_1/results/test_metrics.csv).

### DLA 2 — adattamento per il sentiment

![Accuracy di test su Rotten Tomatoes](DLA_2/figures/sentiment_test_accuracy.svg)

Il fine-tuning completo ha raggiunto la maggiore accuracy di test, ma LoRA ne ha conservato quasi interamente il valore ottimizzando circa l'1.09% dei parametri del modello. Fonte: [`DLA_2/results/sentiment_results.csv`](DLA_2/results/sentiment_results.csv).

### DLA 3 — valutazione LunarLander

![Analisi della temperatura della policy LunarLander](DLA_3/figures/lunarlander_temperature_sweep.svg)

Le ampie deviazioni standard mostrano perché un singolo rollout o il return di training non sono stati usati per la selezione del modello. La temperatura `0.75` è stata scelta tramite un punteggio di affidabilità, non perché massimizzasse soltanto il return medio. Fonte: [`DLA_3/results/lunarlander_temperature_sweep.csv`](DLA_3/results/lunarlander_temperature_sweep.csv).

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
│   └── build_report_assets.py
├── DLA_1/
│   ├── DLA_1.ipynb
│   ├── ASSIGNMENT.md
│   ├── README.md
│   ├── config/  figures/  results/
│   ├── notebooks/  scripts/  src/
│   └── exploratory/
├── DLA_2/
│   ├── DLA_2.ipynb
│   ├── ASSIGNMENT.md
│   ├── README.md
│   └── config/  figures/  results/  notebooks/  scripts/  src/
└── DLA_3/
    ├── DLA_3.ipynb
    ├── ASSIGNMENT.md
    ├── README.md
    ├── config/  figures/  results/
    ├── notebooks/  scripts/  src/
    └── exploratory/
```

Dataset, checkpoint dei modelli, run W&B, directory di output Hugging Face, cache e artefatti locali completi sono deliberatamente esclusi da questo albero e da Git.

## Come esaminare la consegna

1. Leggere questa panoramica, quindi aprire il README dettagliato di ciascun laboratorio.
2. Aprire i tre notebook-indice principali: [`DLA_1.ipynb`](DLA_1/DLA_1.ipynb), [`DLA_2.ipynb`](DLA_2/DLA_2.ipynb) e [`DLA_3.ipynb`](DLA_3/DLA_3.ipynb).
3. Seguire i notebook dettagliati nell'ordine degli esercizi indicato in ciascun `notebooks/README.md`.
4. Esaminare gli output preservati nei notebook e le evidenze leggere nella relativa directory `results/`.
5. Considerare `exploratory/` soltanto come contesto storico: non fa parte del percorso di esecuzione finale.

Training lunghi, estrazione delle feature, rivalutazione finale sul test, logging W&B e run LunarLander sono disattivati per impostazione predefinita quando è esposto un flag `RUN_*` o `ENABLE_WANDB`. Gli output esistenti sono conservati per la revisione; l'attivazione di un flag costituisce una richiesta esplicita di ricalcolo dell'esperimento.

## Installazione ed esecuzione

`requirements.txt` è la lista canonica e multipiattaforma delle dipendenze. Include anche l'installazione editable della repository, così i package `dla_lab1`, `dla_lab2` e `dla_lab3` sono importabili senza modificare `sys.path`. `environment.yml` crea un ambiente Conda e delega l'installazione dei pacchetti allo stesso file. `pyproject.toml` dichiara soltanto i metadati minimi dei package locali e la configurazione di formatter e linter: non introduce una lista di dipendenze alternativa.

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

Linux/WSL è consigliato per DLA 3 perché le dipendenze Box2D e di rendering richieste da LunarLander sono più affidabili in quell'ambiente. CUDA è facoltativo; l'esecuzione su CPU è supportata, ma l'estrazione completa delle feature e il training sono sensibilmente più lenti.

## Riproducibilità e artefatti

- I seed globali sono `42` per DLA 1 e DLA 2 e `2112` per DLA 3. I seed degli ambienti e delle valutazioni sono derivati deterministicamente nell'implementazione.
- I download dei dataset non sono versionati. Le relazioni registrano le dimensioni esatte degli split osservate nei notebook eseguiti.
- I checkpoint (`*.pt`, `*.safetensors` e cartelle checkpoint di Hugging Face) restano locali a causa delle dimensioni.
- Metriche finali leggere, cronologie e input dei grafici sono versionati in `results/`.
- [`tools/build_report_assets.py`](tools/build_report_assets.py) rigenera gli SVG delle relazioni ed estrae output PNG selezionati dai notebook versionati senza ripetere il training.
- La consultazione rapida usa gli output salvati nei notebook e `results/`; l'esecuzione completa richiede dataset, download dei modelli, eventuali credenziali W&B e l'attivazione esplicita dei flag di training.
- La repository non dichiara una riproducibilità bit per bit tra versioni differenti di CUDA, driver, PyTorch o Gymnasium. Per questo la valutazione stocastica di RL è riportata su più episodi.

## Uso dell'IA e integrità accademica

ChatGPT e OpenAI Codex hanno supportato chiarimenti concettuali, debugging, organizzazione del codice, documentazione e controlli di coerenza. L'autore ha eseguito e revisionato gli esperimenti; gli output dell'IA non sono stati accettati come evidenza sperimentale. La dichiarazione completa è in [`AI_USAGE.md`](AI_USAGE.md). Le regole di condotta e la responsabilità autoriale sono indicate in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Riferimenti

- [Documentazione PyTorch](https://pytorch.org/docs/stable/index.html)
- [Documentazione Torchvision](https://pytorch.org/vision/stable/index.html)
- [Documentazione Hugging Face](https://huggingface.co/docs)
- [Documentazione Gymnasium](https://gymnasium.farama.org/)
- [Documentazione Weights & Biases](https://docs.wandb.ai/)

I riferimenti specifici a dataset, articoli, API e implementazioni sono elencati nelle relazioni dettagliate dei singoli laboratori.
