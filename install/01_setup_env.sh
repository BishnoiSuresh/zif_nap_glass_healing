#!/bin/bash
# =============================================================================
# STEP 1  python environment + LAMMPS source (login node, needs internet)
#   bash install/01_setup_env.sh 2>&1 | tee 01_setup.log
#
#   * conda env from environment.yml (python 3.11, torch with CUDA, mace-torch, ase ...)
#   * ML-IAP extras: cython 3.0.11, cupy, cuEquivariance (matched to the torch CUDA version)
#   * LAMMPS $LAMMPS_TAG (shallow clone)
#   * SevenNet GPU-D3 (3 source files) patched into the LAMMPS cmake build. D3 is
#     OPTIONAL: the composite/healing results of the paper use MACE without D3; D3 was
#     only used for the free-box 40NaP Tg run (POT=A).
# =============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/site.sh
D3URL=https://raw.githubusercontent.com/MDIL-SNU/SevenNet/main/sevenn/pair_e3gnn
type module >/dev/null 2>&1 && { module purge; module load anaconda/2025.06 || true; }
source "$(conda info --base)/etc/profile.d/conda.sh"
mkdir -p $SW

# ---------------------------------------------------------------- conda env
if [ ! -d "$ENV" ]; then
  conda env create -p $ENV -f $ROOT/environment.yml
fi
conda activate $ENV

TCUDA=$(python -c "import torch;print(torch.version.cuda or '')")
echo "torch $(python -c 'import torch;print(torch.__version__)') built for CUDA $TCUDA"
case "$TCUDA" in
  12.*) CUPY=cupy-cuda12x ; CUEQ_OPS=cuequivariance-ops-torch-cu12 ;;
  13.*) CUPY=cupy-cuda13x ; CUEQ_OPS=cuequivariance-ops-torch-cu13 ;;
  *)    echo "torch in $ENV has no CUDA build -- install a CUDA build of torch first"; exit 1 ;;
esac

# pin the installed torch so pip never replaces it
TV=$(python -c "import torch;print(torch.__version__)")
echo "torch==$TV" > $SW/pip_constraints.txt
pip install -c $SW/pip_constraints.txt --upgrade-strategy only-if-needed \
            "cython==3.0.11" $CUPY cuequivariance-torch $CUEQ_OPS "cmake>=3.20" ninja
python - <<'PY'
import torch, mace, cupy, Cython, numpy
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| mace", mace.__version__,
      "| cupy", cupy.__version__, "| cython", Cython.__version__, "| numpy", numpy.__version__)
import cuequivariance_torch; print("cuequivariance OK")
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE  # noqa: F401
print("mace mliap exporter OK")
PY

# ---------------------------------------------------------------- LAMMPS (shallow)
cd $SW
if [ ! -d lammps ]; then
  git clone --depth 1 --branch $LAMMPS_TAG https://github.com/lammps/lammps.git lammps
fi
grep LAMMPS_VERSION lammps/src/version.h

# ---------------------------------------------------------------- GPU-D3 (optional)
L=$SW/lammps
if [ "${WITH_D3:-1}" = 1 ] && [ ! -f $L/src/pair_d3.cu ]; then
  for f in pair_d3.cu pair_d3.h pair_d3_pars.h; do curl -fsSL $D3URL/$f -o $L/src/$f; done
  cp $L/cmake/CMakeLists.txt $L/cmake/CMakeLists.txt.orig
  sed -i 's/LANGUAGES CXX C)/LANGUAGES CXX C CUDA)/' $L/cmake/CMakeLists.txt
  sed -i 's|file(GLOB ALL_SOURCES CONFIGURE_DEPENDS ${LAMMPS_SOURCE_DIR}/\[^.\]\*.cpp)|file(GLOB ALL_SOURCES CONFIGURE_DEPENDS ${LAMMPS_SOURCE_DIR}/[^.]*.cpp ${LAMMPS_SOURCE_DIR}/[^.]*.cu)|' $L/cmake/CMakeLists.txt
  cat >> $L/cmake/CMakeLists.txt <<'EOF'

# ---- SevenNet GPU-D3 (pair_style d3) ----
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fmad=false -O3 --expt-relaxed-constexpr")
target_link_libraries(lammps PUBLIC CUDA::cudart CUDA::cuda_driver)
EOF
  grep -q "LANGUAGES CXX C CUDA)" $L/cmake/CMakeLists.txt && grep -q '\[^.\]\*.cu)' $L/cmake/CMakeLists.txt \
     && echo "D3 patch applied" || { echo "D3 patch FAILED"; exit 1; }
fi
echo "Done. Next: install/02_build_lammps.sh on a GPU node."
