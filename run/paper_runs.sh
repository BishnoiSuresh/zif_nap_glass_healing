#!/bin/bash
# =============================================================================
# paper_runs.sh -- the runs behind the paper's MD results, in order (PBS syntax; each
# line is one GPU job). Wall times are for one NVIDIA A100 (FP64 model).
# Submit each block only after the previous block has finished.
#
#   export REPO=/path/to/this/repo ; mkdir -p run1 && cd run1 ; then copy the lines below.
# For SLURM use:  sbatch --export=ALL,REPO=$REPO,STEP=...  $REPO/run/slurm_job.sh
# =============================================================================
J=$REPO/run/pbs_job.pbs

# --- 1. parent glasses ------------------------------------------------------------------
qsub -N zif     -l walltime=48:00:00 -v REPO=$REPO,STEP=zif                  $J   # ZIF-62 glass layer, ~12-15 h
qsub -N zif222  -l walltime=72:00:00 -v REPO=$REPO,STEP=zif,CELL=222         $J   # ZIF-62 T_g run (2368 at.), ~1-2 days
qsub -N napIA   -l walltime=24:00:00 -v REPO=$REPO,STEP=nap,MODE=iso,POT=A   $J   # 40NaP Tg (MH-1 + D3, 0 bar), ~9 h
qsub -N napIB   -l walltime=24:00:00 -v REPO=$REPO,STEP=nap,MODE=iso,POT=B   $J   # 40NaP for XRD (no D3, 1 GPa), ~8 h
# after zif has finished (needs the ZIF lateral box):
qsub -N napLB   -l walltime=24:00:00 -v REPO=$REPO,STEP=nap,MODE=layer,POT=B $J   # 40NaP layer, ~8 h

# --- 2. composite -----------------------------------------------------------------------
qsub -N comp    -l walltime=48:00:00 -v REPO=$REPO,STEP=composite            $J   # ~17 h

# --- 3. crack + healing: 4 temperatures x 3 seeds --------------------------------------
for S in 1 2 3; do
  qsub -N h500_$S -l walltime=48:00:00 -v REPO=$REPO,STEP=heal,THEAL=500,THOLD=200,SEED=$S $J
  qsub -N h600_$S -l walltime=48:00:00 -v REPO=$REPO,STEP=heal,THEAL=600,THOLD=200,SEED=$S $J
  qsub -N h700_$S -l walltime=24:00:00 -v REPO=$REPO,STEP=heal,THEAL=700,THOLD=100,SEED=$S $J
  qsub -N h800_$S -l walltime=24:00:00 -v REPO=$REPO,STEP=heal,THEAL=800,THOLD=100,SEED=$S $J
done

# --- 4. elastic tensors -----------------------------------------------------------------
qsub -N el0 -l walltime=24:00:00 -v REPO=$REPO,STEP=elastic,ITEMS=pristine+cracked+contact $J
for T in 500 600 700 800; do
  qsub -N el$T -l walltime=24:00:00 \
       -v REPO=$REPO,STEP=elastic,ITEMS=heal_B_T${T}_s1+heal_B_T${T}_s2+heal_B_T${T}_s3 $J
done

# --- optional (not in the paper) ---------------------------------------------------------
# 0-bar healing:     -v REPO=$REPO,STEP=heal,THEAL=800,SEED=1,PHEAL=0,SUF=_p0
# uniaxial tension:  -v REPO=$REPO,STEP=tension,ITEMS=pristine+heal_B_T800_s1
