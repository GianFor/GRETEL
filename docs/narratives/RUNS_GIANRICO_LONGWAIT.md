# B1 completa su H200 con attesa lunga

Il 14 novembre 2026 `scontrol show partition cuda` riporta
`MaxTime=10-00:00:00`. La variante `matrix_20b_h200_longwait.json` conserva
modelli e probe della matrice H200 4k con OSS-20B. Imposta il budget vLLM al
60% (circa 84 GiB sulla H200 osservata), il tempo massimo del job a 10 giorni
e l'attesa della VRAM a 691200 secondi (8 giorni). Il profilo al 60% deve ancora
essere verificato in esecuzione, soprattutto per i modelli grandi.
Eventuali limiti di account/QOS restano soggetti alla verifica di Slurm.

Ogni task aspetta almeno 92000 MiB liberi su `compute-2-3`, controllando ogni
5 minuti. L'attesa della VRAM consuma il limite del job; l'attesa in coda no.
Restano circa due giorni per caricamento e inferenza dopo l'attesa massima.
Non è una prenotazione esclusiva. I limiti sono per task, non per tutta B1.

La configurazione è versionata nel branch `llm-probes-consistency`.
Il blocco seguente aggiorna il checkout e usa direttamente il file tracciato.
Usa direttamente le coppie originali di Rodrigo: il percorso
`narr-tcr64_dce_lclsf` è anche quello selezionato dai filtri di sweep e
autovalutazione. Una copia da `narr-matrix/tcr64-20b/source` perde quel nome
nel manifest preparato e non va usata con questi filtri invariati.

Eseguire una sola volta da Caliban. Il terminale aspetta la fine del prepare
CPU, poi sottomette i due array: 5 narratori e 8 valutatori, un task alla volta.
I job della precedente run 9B/20B non vengono modificati né annullati.

```bash
cd ~/GRETEL && bash <<'BASH'
set -euo pipefail
git pull --ff-only origin llm-probes-consistency
export GRETEL_ENV=GRTL WAIT_STEP=300 SKIP_GPU_WAIT=0
run=lab/output/results/narr-matrix/tcr64-20b-h200-longwait
source=lab/output/results/narr-tcr64_dce_lclsf
config=lab/config/narratives/matrix_20b_h200_longwait.json
if [ -e "$run" ]; then
  echo "Run già presente: $run. Mi fermo per evitare duplicati."
  exit 1
fi
test -d "$source"
test -f "$config"
mkdir -p lab/output/logs
sbatch --wait -p normal scripts/slurm_matrix_prepare.sh "$config" "$run" "$source"
python3 scripts/paper_probe_matrix.py fleet --max 1 --run-root "$run"
BASH
```

Il dry run locale verifica i limiti e gli array `0-4%1` e `0-7%1`.
La dipendenza dei valutatori è `afterany`: aspettano la fine di tutti i
generatori, anche se alcuni sono falliti. I task dello stesso array proseguono
indipendentemente. Un giudice esterno registra un errore per un generatore
incompleto e continua sugli altri; può terminare con codice 2 pur avendo
salvato valutazioni utili. L'autovalutazione del generatore incompleto può
fallire senza bloccare gli altri task.

La modifica del JSON sorgente non aggiorna manifest, configurazioni congelate
o parametri Slurm di una run già preparata o accodata. Prima di sostituire
una sottomissione esistente, controllarne gli ID e gli stati; non rilanciare
il blocco come se aggiornasse i job precedenti.

Non è stata effettuata una sottomissione da questa postazione. Conservare gli
ID stampati; controllare copertura, errori e parsing nei risultati anche quando
i job terminano con `COMPLETED`. Se una sottomissione fallisce, controllare
`jobs.jsonl` prima di riprovare per evitare duplicati.
