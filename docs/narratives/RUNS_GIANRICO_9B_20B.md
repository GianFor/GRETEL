# Tree-Cycles · Qwen 9B e GPT-OSS-20B su H200

Prima parte della matrice concordata, sui 100 dump Tree-Cycles già congelati. Usa Qwen3.5-9B per generare e GPT-OSS-20B per i probe Reconstruction, Consistency e Recourse definiti nella matrice H200. Gli altri narratori, GLM, lo sweep Recourse e l'autovalutazione restano da eseguire.

Il budget vLLM è 0.30 della memoria totale della H200, circa 42 GiB, con una sequenza simultanea. Il precedente smoke Qwen 9B aveva generato con circa 40 GiB di budget. La valutazione GPT-OSS-20B con questi prompt e questo budget richiede ancora verifica sul nodo. Il limite di output è 4.096 token, il contesto massimo 40.960. Il launcher attende fino a 30 minuti per almeno 45.000 MiB liberi, controllando ogni minuto.

## Lancio da Caliban

```bash
cd "$HOME/GRETEL" &&
git pull --ff-only origin llm-probes-consistency &&
bash scripts/launch_tcr_9b_20b_h200.sh
```

Esegui il comando una volta. Ogni invocazione crea una nuova run e invia tre job con dipendenze `afterok`. Il prepare riusa `tcr64-20b/source`. Il generatore parte dopo il prepare, il giudice dopo il generatore. Un predecessore fallito causa l'annullamento del job dipendente. Il lancio stampa subito gli ID e li registra nella directory `Submission`, anche se una sottomissione successiva fallisce.

## Stato e risultati

Usa il comando `sacct` stampato dal launcher. Per seguire la generazione, usa `tail -n 40 -f` sul percorso `Generator log` stampato. Le righe `instance` indicano avanzamento sui grafi. `Ctrl+C` interrompe il monitor. La coda personale si vede con `squeue -u "$USER"`.

Imposta `run` al percorso stampato dal launcher, poi genera il riepilogo.

```bash
conda activate GRTL
python scripts/paper_probe_matrix.py summary --run-root "$run"
```

Per Rodrigo raccogli `matrix-summary.csv`, `generated/qwen3.5-9b/generation.json` e l'output di `sacct`. Verifica gli stati di parsing e la copertura anche quando Slurm riporta `COMPLETED`. Il riepilogo può essere generato durante la run per controllare risultati parziali.

Gli ID di questa sottomissione sono in `Submission/jobs.txt`. Questo launcher usa sottomissioni dirette e registra i job in quel file. Per le run create da questo launcher, usa i tre ID stampati per monitorare e riprovare eventuali job falliti.
