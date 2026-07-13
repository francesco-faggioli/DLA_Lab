# Uso di strumenti assistiti dall'IA

## Strumenti e ambito

ChatGPT e OpenAI Codex sono stati usati come strumenti di supporto durante i tre laboratori e la revisione finale della repository. L'assistenza ha riguardato:

- chiarimenti concettuali su deep learning e reinforcement learning;
- interpretazione e scomposizione delle consegne ufficiali;
- debugging e revisione del codice Python e dei notebook;
- organizzazione di funzioni, moduli, script e configurazioni riutilizzabili;
- commenti e docstring;
- diagnosi di problemi relativi a percorsi, ambiente, dipendenze e stato dei notebook;
- progettazione e revisione della struttura dei README;
- organizzazione della repository e controllo di nomi, link e coerenza Git;
- verifica della tracciabilità delle metriche riportate rispetto agli output dei notebook o agli artefatti salvati;
- discussione degli iperparametri A2C e delle scelte di valutazione in DLA 3.

L'assistenza dell'IA non è stata considerata uno strumento sperimentale. Una spiegazione generata o un valore proposto non sono stati accettati come evidenza finché non sono stati verificati rispetto al codice, a un output eseguito, a una configurazione, a una metrica salvata o a una fonte esterna.

## Uso specifico nei laboratori

### DLA 1

Il supporto dell'IA è stato usato per organizzare la pipeline GTSRB, separare l'estrazione delle feature dal fine-tuning, revisionare le metriche di retrieval, migliorare commenti e documentazione e controllare la coerenza tra artefatti locali e riepiloghi W&B. Weights & Biases è stato usato come servizio di tracciamento per alcune run e non è un sistema di IA. Il suo ruolo pubblico in questa repository è documentato tramite cronologie e riepiloghi esportati, poiché le directory W&B locali grezze non sono versionate.

### DLA 2

Il supporto dell'IA è stato usato per revisionare il codice di tokenizzazione DistilBERT e Trainer, strutturare il confronto tra LoRA e congelamento parziale, risolvere problemi dell'ambiente locale Hugging Face e documentare l'esperimento CLIP-Adapter. I valori finali di accuracy, F1, precision, recall, loss e numero di parametri provengono dai notebook eseguiti.

### DLA 3

Il supporto dell'IA è stato usato per discutere la varianza di REINFORCE, la value baseline appresa, gli aggiornamenti A2C, il disegno della valutazione e gli iperparametri LunarLander. Il riferimento esterno esatto per i parametri è stata la [configurazione A2C di RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml). È stata usata come punto di partenza, non copiata come soluzione completa. L'implementazione locale, i budget di training, le varianti dei checkpoint, le modalità della policy, le temperature e la selezione finale sono stati revisionati e valutati sperimentalmente.

I suggerimenti dell'IA non sono stati accettati automaticamente. In particolare, la configurazione finale LunarLander è stata scelta tramite valutazioni ripetute dei checkpoint e delle policy, mentre i raffinamenti non riusciti restano documentati.

## Verifica e responsabilità umana

L'autore:

- ha eseguito gli esperimenti rappresentati dagli output preservati nei notebook;
- ha revisionato il codice e le spiegazioni generati con assistenza dell'IA;
- ha verificato le metriche finali rispetto ai notebook e alle evidenze salvate;
- ha mantenuto la responsabilità di tutte le decisioni scientifiche e ingegneristiche;
- resta responsabile della capacità di spiegare codice, metodi, risultati e limiti.

Nessuna metrica, figura, iperparametro, fonte o esercizio completato è stato inventato per migliorare la presentazione. La revisione finale della repository non ha modificato algoritmi, seed degli esperimenti finali, split dei dataset, impostazioni di ottimizzazione o risultati sperimentali principali. Nuove liste di seed sono state introdotte soltanto per i controlli visuali qualitativi richiesti, composti da cinque episodi CartPole e LunarLander; i relativi riepiloghi eseguiti direttamente sono versionati separatamente e non vengono trattati come benchmark.

## Limiti dell'assistenza dell'IA

I sistemi di IA possono produrre codice errato, spiegazioni scientifiche non supportate, riferimenti inventati e suggerimenti di parametri inappropriati. I loro output richiedono una verifica indipendente. Questa repository collega quindi direttamente le fonti esterne e mantiene visibili incertezze, risultati negativi e lacune reali nelle evidenze.

## Fonti esterne consultate

| Fonte | Uso | Ruolo nella verifica |
| --- | --- | --- |
| [Documentazione PyTorch](https://pytorch.org/docs/stable/index.html) | Tensori, modelli, ottimizzazione, distribuzioni | Riferimento per API e concetti |
| [Documentazione Torchvision](https://pytorch.org/vision/stable/index.html) | GTSRB e modelli ResNet pre-addestrati | Riferimento per API di dataset e modelli |
| [Documentazione Weights & Biases](https://docs.wandb.ai/) | Tracciamento facoltativo degli esperimenti | Riferimento per logging e flusso degli artefatti |
| [Scheda del dataset Rotten Tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | Struttura e provenienza del dataset | Riferimento per split ed etichette |
| [Documentazione DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) | Comportamento di tokenizer e modello | Riferimento per le API Transformer |
| [Documentazione PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora) | Configurazione LoRA | Riferimento per il fine-tuning efficiente |
| [Repository OpenCLIP](https://github.com/mlfoundations/open_clip) | Caricamento e preprocessing CLIP | Riferimento per le API di implementazione |
| [Repository ImageNet-Sketch](https://github.com/HaohanWang/ImageNet-Sketch) | Dataset di immagini fuori dominio | Riferimento per dataset e articolo |
| [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) | Specifica dell'ambiente | Riferimento per stato, azioni, reward e terminazione |
| [Gymnasium LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/) | Specifica dell'ambiente | Riferimento per stato, azioni, reward e successo |
| [YAML A2C di RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml) | Discussione degli iperparametri DLA 3 | Termine di confronto esterno |

## Dichiarazione di integrità accademica

Gli strumenti di IA hanno assistito il lavoro, ma non sostituiscono l'autorialità, la comprensione o la responsabilità. L'autore è responsabile della consegna e della spiegazione di ogni sua parte. Si veda anche [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
