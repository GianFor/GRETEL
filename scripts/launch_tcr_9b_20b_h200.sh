#!/usr/bin/env bash
# Submit prepare, Qwen9B generation and GPT-OSS-20B evaluation in sequence.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

config=lab/config/narratives/matrix_9b_20b_h200_4k.json
source_run=lab/output/results/narr-matrix/tcr64-20b/source
test -f "$config"
test -d "$source_run"
command -v sbatch >/dev/null
mkdir -p lab/output/logs
if [ "$#" -eq 2 ] && [ "$1" = --replace-pending ]; then
  submission=$(cd "$2" && pwd)
  run=$(cat "$submission/run.txt")
  prepare_id=$(awk '$1 == "prepare" {id=$2} END {print id}' "$submission/jobs.txt")
  old_generator=$(awk '$1 == "generator" {id=$2} END {print id}' "$submission/jobs.txt")
  old_judge=$(awk '$1 == "judge" {id=$2} END {print id}' "$submission/jobs.txt")
  for job in "$prepare_id" "$old_generator" "$old_judge"; do
    [[ "$job" =~ ^[0-9]+$ ]] || { echo 'Missing job ID in jobs.txt'; exit 1; }
  done
  # Holding pending jobs closes the race between the state check and cancellation.
  # A running job can also be held by Slurm, so verify the state after holding.
  trap 'scontrol release "$old_generator" "$old_judge" >/dev/null 2>&1 || true' EXIT
  scontrol hold "$old_generator" "$old_judge"
  for job in "$old_generator" "$old_judge"; do
    if [ "$(squeue -h -j "$job" -o '%T')" != PENDING ]; then
      echo "Job $job is no longer pending. Check sacct before replacing it."
      exit 1
    fi
  done
  scancel "$old_judge" "$old_generator"
  trap - EXIT
  printf 'replaced %s %s\n' "$old_generator" "$old_judge" | tee -a "$submission/jobs.txt"
elif [ "$#" -eq 0 ]; then
  submission=$(mktemp -d "$PWD/lab/output/logs/tcr-9b-20b-h200-XXXXXX")
  run="$PWD/lab/output/results/narr-matrix/$(basename "$submission")"
  cp "$config" "$submission/config.json"
  printf '%s\n' "$run" > "$submission/run.txt"
  printf 'Submission %s\nRun %s\n' "$submission" "$run"

  # Record each successful submission immediately, including partial launches.
  prepare_id=$(sbatch --parsable -p normal --export=ALL,GRETEL_ENV=GRTL \
    scripts/slurm_matrix_prepare.sh "$submission/config.json" "$run" "$source_run")
  prepare_id=${prepare_id%%;*}
  printf 'prepare %s\n' "$prepare_id" | tee -a "$submission/jobs.txt"
else
  echo 'Usage: bash scripts/launch_tcr_9b_20b_h200.sh [--replace-pending SUBMISSION_DIRECTORY]'
  exit 2
fi

printf 'GPU wait limit: 43200 seconds; job time limit: 24 hours\n' | tee -a "$submission/jobs.txt"

gpu_options=(
  --parsable -p cuda --nodelist=compute-2-3 --gres=gpu:fast:1
  --mem=96G --time=1-00:00:00 --kill-on-invalid-dep=yes
  --export=ALL,MIN_FREE_MB=45000,MAX_WAIT=43200,WAIT_STEP=60,SKIP_GPU_WAIT=0,GRETEL_ENV=GRTL-llm,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1
)
generator_id=$(sbatch "${gpu_options[@]}" --job-name=tcr-qwen9b \
  --dependency="afterok:$prepare_id" \
  scripts/slurm_paper_probes.sh matrix-generator "$run" qwen3.5-9b)
generator_id=${generator_id%%;*}
printf 'generator %s\n' "$generator_id" | tee -a "$submission/jobs.txt"

judge_id=$(sbatch "${gpu_options[@]}" --job-name=tcr-oss20b \
  --dependency="afterok:$generator_id" \
  scripts/slurm_paper_probes.sh matrix-judge "$run" gpt-oss-20b)
judge_id=${judge_id%%;*}
printf 'judge %s\n' "$judge_id" | tee -a "$submission/jobs.txt"

printf '\nCheck: sacct -X -j %s,%s,%s --format=JobID,State,Elapsed,ExitCode\n' \
  "$prepare_id" "$generator_id" "$judge_id"
printf 'Generator log: lab/output/logs/slurm_%s.out\n' "$generator_id"
printf 'Judge log: lab/output/logs/slurm_%s.out\n' "$judge_id"
