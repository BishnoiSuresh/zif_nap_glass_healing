#!/bin/bash
#SBATCH --job-name=zifnap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
##SBATCH --account=<your_account>
##SBATCH --partition=<gpu_partition>
# -----------------------------------------------------------------------------
# SLURM wrapper for run/run_step.sh. Submit from your RUN directory, e.g.
#   sbatch --export=ALL,REPO=/path/to/repo,STEP=heal,THEAL=800,SEED=1 /path/to/repo/run/slurm_job.sh
# -----------------------------------------------------------------------------
[ -n "$REPO" ] || { echo "pass REPO=/path/to/repo with --export"; exit 2; }
cd ${SLURM_SUBMIT_DIR}
bash $REPO/run/run_step.sh
