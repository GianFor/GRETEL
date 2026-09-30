#!/bin/bash -l
# Same assigned-GPU waiting policy as slurm_probes.sh; paper runner is explicit.
#SBATCH -J paper-probes
#SBATCH -p cuda
#SBATCH --gres=gpu:fast:1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o lab/output/logs/slurm_%j.out
set -euo pipefail

pass=${1:?usage: generate CONFIG | probe CONFIG RESULTS [PROBE]}
config=${2:?missing config}
source ~/miniforge3/etc/profile.d/conda.sh
conda activate GRTL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}

gpu=${CUDA_VISIBLE_DEVICES:-0}
gpu=${gpu%%,*}
waited=0
while true; do
  free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  echo "GPU $gpu: $free_mb MiB free"
  [ "$free_mb" -ge "${MIN_FREE_MB:-30000}" ] && break
  [ "$waited" -lt "${MAX_WAIT:-21600}" ] || exit 3
  sleep "${WAIT_STEP:-300}"
  waited=$((waited + ${WAIT_STEP:-300}))
done

case "$pass" in
  generate) python main.py "$config" "${RUN_ID:-1}" ;;
  probe)
    results=${3:?missing results root}
    extra=()
    [ -z "${4:-}" ] || extra=(--probe "$4")
    python scripts/run_paper_probes.py --config "$config" --results "$results" --limit "${SAMPLE_LIMIT:-5}" "${extra[@]}"
    ;;
  *) echo "unknown pass: $pass"; exit 2 ;;
esac
