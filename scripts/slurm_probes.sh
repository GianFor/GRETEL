#!/bin/bash -l
# One Slurm job for either pass of the probes protocol. From the repo root:
#
#   sbatch scripts/slurm_probes.sh generate lab/config/probes/tcr-100_dce-lcls_qwen3-8b.jsonc
#   sbatch scripts/slurm_probes.sh probe    lab/config/probes/judge_gpt-oss-20b.jsonc lab/output/results/probes-pilot
#
# Site settings (partition, GPU class, conda path) are at the top; adjust them
# to the cluster at hand. The job picks the GPU with the most free memory at
# start, since GPUs may be shared, and refuses to run under MIN_FREE_MB.
#SBATCH -J probes
#SBATCH -p cuda
#SBATCH --gres=gpu:fast:1
#SBATCH -c 8
#SBATCH --time=12:00:00
#SBATCH -o lab/output/logs/slurm_%j.out

CONDA_ENV=GRTL
MIN_FREE_MB=40000

pass=$1; config=$2; results=$3
if [ -z "$pass" ] || [ -z "$config" ]; then
  echo "usage: $0 generate <config> | probe <config> <results_root>"; exit 2
fi

source ~/miniforge3/etc/profile.d/conda.sh && conda activate $CONDA_ENV
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export HF_HOME=${HF_HOME:-$HOME/hf_cache}
mkdir -p lab/output/logs

# Freest GPU first; abort if even that one is too full
gpu=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1)
export CUDA_VISIBLE_DEVICES=${gpu%%,*}
free_mb=${gpu##*,}
echo "GPU $CUDA_VISIBLE_DEVICES with ${free_mb} MiB free"
if [ "${free_mb// /}" -lt "$MIN_FREE_MB" ]; then
  echo "less than ${MIN_FREE_MB} MiB free, giving up"; exit 3
fi

case $pass in
  generate) python main.py "$config" 1 ;;
  probe)    python scripts/run_probes.py --config "$config" --results "${results:-lab/output/results}" ;;
  *)        echo "unknown pass: $pass"; exit 2 ;;
esac
