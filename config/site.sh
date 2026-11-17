# =============================================================================
# config/site.sh -- the ONLY file you should need to edit for your machine.
# Sourced by install/*.sh and run/run_step.sh.  Every value can also be overridden
# from the environment (e.g.  SW=/scratch/me/lmp bash install/01_setup_env.sh).
# =============================================================================

# where LAMMPS, the exported model and the build live
SW=${SW:-$HOME/software/lammps-mliap}
# conda environment (python + torch + mace + the LAMMPS python module)
ENV=${ENV:-$HOME/envs/mace-lammps}
# LAMMPS executable and the ML-IAP model used for every production run
LMP=${LMP:-$SW/lammps/build/lmp}
MODEL=${MODEL:-$SW/models/mh1_matpes_r2scan_fp64-mliap_lammps.pt}

# LAMMPS version used for the paper ("LAMMPS (4 Jul 2026)")
LAMMPS_TAG=${LAMMPS_TAG:-patch_4Jul2026}
# GPU architecture for Kokkos / nvcc (A100 = AMPERE80 / 80; H100 = HOPPER90 / 90)
KOKKOS_ARCH=${KOKKOS_ARCH:-AMPERE80}
CUDA_ARCH=${CUDA_ARCH:-80}
NJ=${NJ:-8}                       # parallel build jobs

# -----------------------------------------------------------------------------
# load_modules: compilers / CUDA / MPI. Edit for your cluster.
# Requirements: GCC >= 10.4 (Kokkos 5 needs C++20), a CUDA toolkit matching the CUDA
# version torch was built for, an MPI, cmake >= 3.20.
# The example below is the HPC setup used for the paper (PBS, A100 GPUs).
# -----------------------------------------------------------------------------
load_modules () {
  if type module >/dev/null 2>&1; then
    module purge
    module load anaconda/2025.06 gcc/10.5.0
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV"
  if type module >/dev/null 2>&1; then
    case "$(python -c 'import torch;print(torch.version.cuda)')" in
      12.*) module load cuda/12.6 openmpi/5.0.9 ;;
      13.*) module load cuda/13.0 openmpi/5.0.9_cuda13 ;;
    esac
  fi
  export LD_LIBRARY_PATH=$SW/lammps/build:${LD_LIBRARY_PATH:-}
}
