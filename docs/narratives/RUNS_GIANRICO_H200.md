# Narrative matrix on H200 with GPT-OSS-20B

Use [matrix_20b_h200_4k.json](../../lab/config/narratives/matrix_20b_h200_4k.json) from the GRETEL checkout on Caliban. The matrix uses GPT-OSS-20B for the primary evaluation and Recourse sweep. The five narrators, GLM judge, probes and self-evaluation follow [the original run plan](RUNS_GIANRICO.md).

The output cap is 4,096 tokens for narrators and self-evaluation. In a one-graph Qwen3.5-9B test, the previous cap of 1,024 tokens cut off five of nine responses. The H200 jobs request `compute-2-3` and check for at least 115,000 MiB of free VRAM. Each allocated task waits up to five minutes for that threshold.

## Prepare B1

The CPU job copies and validates the existing Tree-Cycles dumps. It uses the previously frozen inputs if they are present in this checkout.

```bash
cd ~/GRETEL
dataset=tcr64
run="lab/output/results/narr-matrix/${dataset}-20b-h200-4k"
frozen_source="lab/output/results/narr-matrix/${dataset}-20b/source"
if [ ! -d "$frozen_source" ]; then
  frozen_source="lab/output/results/narr-${dataset}_dce_lclsf"
fi
sbatch -p normal scripts/slurm_matrix_prepare.sh \
  lab/config/narratives/matrix_20b_h200_4k.json "$run" "$frozen_source"
```

After the prepare job completes, its log `lab/output/logs/matrix_prepare_<job>.out` ends with the path of the prepared matrix.

## Launch B1

The dry run should show two arrays with `--nodelist=compute-2-3` and a concurrency limit of one. Submit the arrays once per prepared run.

```bash
cd ~/GRETEL
conda activate GRTL
dataset=tcr64
run="lab/output/results/narr-matrix/${dataset}-20b-h200-4k"
test -f "$run/matrix.json" &&
test ! -e "$run/jobs.jsonl" &&
python scripts/paper_probe_matrix.py fleet --max 1 --run-root "$run" --dry-run
```

```bash
test ! -e "$run/jobs.jsonl" &&
python scripts/paper_probe_matrix.py fleet --max 1 --run-root "$run"
```

Record the two array IDs. Once B1 finishes, generate the summary with `python scripts/paper_probe_matrix.py summary --run-root "$run"` and check the per-task exit codes and the generation statuses. Follow the same steps with `bbbp` and then `aids` after reviewing the previous batch.
