#!/bin/bash -l
# One Slurm job for either pass of the probes protocol. From the repo root:
#
#   sbatch scripts/slurm_probes.sh generate lab/config/probes/tcr-100_dce-lcls_qwen3-8b.jsonc
#   sbatch scripts/slurm_probes.sh probe    lab/config/probes/judge_gpt-oss-20b.jsonc lab/output/results/probes-pilot
#
# Site settings (partition, GPU class, conda path) are at the top; adjust them
# to the cluster at hand. GPUs may be shared with other jobs, so the job waits
# until the GPU Slurm assigned has MIN_FREE_MB free, checking every WAIT_STEP
# seconds for at most MAX_WAIT seconds. The log directory must exist before
# sbatch: mkdir -p lab/output/logs
#SBATCH -J probes
#SBATCH -p cuda
#SBATCH --gres=gpu:fast:1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o lab/output/logs/slurm_%j.out

CONDA_ENV=GRTL
MIN_FREE_MB=${MIN_FREE_MB:-30000}
WAIT_STEP=${WAIT_STEP:-300}
MAX_WAIT=${MAX_WAIT:-21600}

pass=$1; config=$2; results=$3
if [ -z "$pass" ] || [ -z "$config" ]; then
  echo "usage: $0 generate <config> | probe <config> <results_root>"; exit 2
fi

source ~/miniforge3/etc/profile.d/conda.sh && conda activate $CONDA_ENV
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}

# Stay on the GPU Slurm assigned (other GPUs on the node belong to other jobs)
gpu=${CUDA_VISIBLE_DEVICES%%,*}
gpu=${gpu:-0}
waited=0
while true; do
  free_mb=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  echo "$(date '+%F %T') GPU $gpu: ${free_mb} MiB free"
  [ "$free_mb" -ge "$MIN_FREE_MB" ] 2>/dev/null && break
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "still under ${MIN_FREE_MB} MiB free after ${MAX_WAIT}s, giving up"; exit 3
  fi
  sleep "$WAIT_STEP"; waited=$((waited + WAIT_STEP))
done

case $pass in
  generate) python main.py "$config" 1 ;;
  probe)    python scripts/run_probes.py --config "$config" --results "${results:-lab/output/results}" ;;
  *)        echo "unknown pass: $pass"; exit 2 ;;
esac
