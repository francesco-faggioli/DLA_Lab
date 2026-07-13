# Uso di strumenti assistiti dall'IA

## Strumenti e ambito

ChatGPT e OpenAI Codex sono stati utilizzati come strumenti di supporto durante i tre laboratori e nella revisione finale della repository. L'assistenza ha riguardato:

- il chiarimento concettuale di argomenti di deep learning e reinforcement learning;
- l'interpretazione e la scomposizione delle tracce ufficiali;
- il debug e la revisione del codice Python e dei notebook;
- l'organizzazione di funzioni riutilizzabili, moduli, script e configurazioni;
- il miglioramento dei commenti e delle docstring del progetto;
- la diagnosi di problemi relativi a percorsi, ambienti, dipendenze e stato dei notebook;
- la struttura della repository, la stesura dei README, i nomi, i collegamenti e i controlli di coerenza Git;
- la presentazione matematica dei metodi utilizzati nel progetto;
- la verifica della coerenza reciproca tra formule, metriche, figure, output dei notebook e artefatti salvati;
- la discussione degli iperparametri di A2C e delle scelte di valutazione in DLA 3.

L'assistenza dell'IA non è stata considerata uno strumento sperimentale. Una spiegazione, una formula, un frammento di codice o un valore proposto dall'IA non sono stati accettati come evidenza senza una verifica rispetto all'implementazione, a un output eseguito, a una configurazione, a una metrica salvata o a una fonte esterna appropriata.

## Uso specifico per laboratorio

### DLA 1

Il supporto dell'IA è stato impiegato per organizzare la pipeline GTSRB, separare l'estrazione delle feature dal fine-tuning, rivedere le metriche di retrieval, migliorare la documentazione e verificare la presentazione della normalizzazione delle immagini, della cross-entropy, della focal loss, della similarità coseno, della Precision@K e del classificatore Nearest-Mean. Weights & Biases è stato utilizzato come servizio di tracciamento per alcune esecuzioni sperimentali; non è un sistema di IA. Le directory W&B locali grezze non sono versionate.

### DLA 2

Il supporto dell'IA è stato impiegato per rivedere la tokenizzazione di DistilBERT e il codice di Trainer, strutturare il confronto tra fine-tuning completo, LoRA e congelamento parziale, eseguire il debug di problemi dell'ambiente Hugging Face locale e documentare l'esperimento CLIP-Adapter. Le spiegazioni matematiche di softmax, cross-entropy, aggiornamenti a basso rango, percentuale di parametri addestrabili, similarità CLIP e adapter residuo sono state verificate rispetto all'implementazione locale.

### DLA 3

Il supporto dell'IA è stato impiegato per discutere la varianza di REINFORCE, la baseline di valore appresa, la Generalized Advantage Estimation, gli aggiornamenti A2C, la regolarizzazione entropica, il campionamento con temperatura, la progettazione della valutazione e gli iperparametri di LunarLander. Il riferimento esterno per i parametri è stato la [configurazione A2C di RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml). È stata usata come punto di partenza, non copiata come soluzione completa.

L'implementazione locale, i budget di addestramento, le varianti dei checkpoint, le modalità della policy, le temperature e la selezione finale sono stati esaminati e valutati sperimentalmente. I perfezionamenti che non hanno avuto successo restano documentati, anziché essere sostituiti da miglioramenti inventati.

## Presentazione matematica

Gli strumenti di IA sono stati utilizzati per migliorare la chiarezza e il posizionamento delle formule matematiche nelle relazioni e nei notebook dei laboratori. Questo uso è stato vincolato dalle seguenti misure di controllo:

- ogni formula è stata verificata rispetto alla funzione, classe, metrica o cella del notebook effettivamente corrispondente;
- riduzioni, segni, coefficienti, normalizzazioni, maschere di bootstrap, soglie e temperature specifici dell'implementazione sono stati documentati, anziché sostituiti con varianti generiche da manuale;
- le formule non sono state utilizzate per modificare algoritmi, iperparametri, checkpoint, output o risultati riportati;
- sono state escluse le formule che non corrispondevano a codice o a un metodo effettivamente utilizzato nel progetto;
- l'autore resta responsabile della comprensione e della spiegazione di tutto il contenuto matematico.

## Verifica umana e responsabilità

L'autore:

- ha eseguito gli esperimenti rappresentati dagli output conservati nei notebook;
- ha revisionato il codice, i commenti, le docstring e le spiegazioni realizzati con l'assistenza dell'IA;
- ha confrontato le metriche finali con i notebook e le evidenze salvate;
- ha verificato che le affermazioni matematiche corrispondessero all'implementazione;
- ha mantenuto la responsabilità di ogni decisione scientifica e ingegneristica;
- resta responsabile della spiegazione del codice, della matematica, dei risultati e dei limiti.

Nessuna metrica, figura, formula, iperparametro, fonte o esercizio completato è stato inventato per migliorare la presentazione. La revisione della documentazione non ha modificato algoritmi, seed degli esperimenti finali, suddivisioni dei dataset, impostazioni di ottimizzazione, checkpoint o risultati principali.

## Limiti dell'assistenza IA

I sistemi di IA possono produrre codice errato, spiegazioni scientifiche non supportate, formule malformate, riferimenti inventati e suggerimenti inappropriati sui parametri. I loro output richiedono una verifica indipendente. Questa repository collega quindi le fonti esterne, conserva le evidenze misurate e mantiene visibili l'incertezza, i risultati negativi e i limiti reali.

## Fonti esterne consultate

| Fonte | Uso | Ruolo nella verifica |
| --- | --- | --- |
| [Documentazione di PyTorch](https://pytorch.org/docs/stable/index.html) | Tensori, modelli, ottimizzazione e distribuzioni di probabilità | Riferimento per API e implementazione |
| [Documentazione di Torchvision](https://pytorch.org/vision/stable/index.html) | GTSRB e modelli ResNet preaddestrati | Riferimento per le API di dataset e modelli |
| [Documentazione di Weights & Biases](https://docs.wandb.ai/) | Tracciamento facoltativo degli esperimenti | Riferimento per logging e gestione degli artefatti |
| [Scheda del dataset Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | Struttura e provenienza del dataset | Riferimento per suddivisioni ed etichette |
| [Documentazione di DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) | Tokenizer e comportamento per la classificazione di sequenze | Riferimento per le API dei Transformer |
| [Documentazione PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora) | Configurazione LoRA | Riferimento per il fine-tuning efficiente |
| [Repository OpenCLIP](https://github.com/mlfoundations/open_clip) | Caricamento e pre-elaborazione CLIP | Riferimento per le API di implementazione |
| [Repository ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) | Dataset di immagini fuori distribuzione | Riferimento per dataset e articolo |
| [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | Specifica dell'ambiente | Riferimento per stato, azione, ricompensa e terminazione |
| [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | Specifica dell'ambiente | Riferimento per stato, azione, ricompensa e successo |
| [YAML A2C di RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml) | Discussione degli iperparametri di DLA 3 | Punto di confronto esterno |

## Dichiarazione di integrità accademica

Gli strumenti di IA hanno assistito il lavoro, ma non sostituiscono la paternità, la comprensione o la responsabilità. L'autore è responsabile della consegna e della spiegazione di ogni sua parte. Si veda anche [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
