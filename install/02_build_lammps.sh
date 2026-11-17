#!/bin/bash
# =============================================================================
# STEP 2  build LAMMPS with KOKKOS (CUDA) + ML-IAP (python, MACE unified) on a GPU node
#   bash install/02_build_lammps.sh 2>&1 | tee 02_build.log
#   (or submit it as a batch job on a GPU node, ~30-60 min)
# =============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/site.sh
set +u; load_modules; set -u

cd $SW/lammps
echo "Node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null) | $(date)"
echo "nvcc: $(nvcc --version | tail -1) | gcc: $(gcc -dumpversion) | cmake: $(cmake --version | head -1)"
GV=$(gcc -dumpversion | cut -d. -f1); [ "$GV" -ge 10 ] || { echo "gcc $GV too old for Kokkos 5 (need >= 10.4)"; exit 1; }

rm -rf build
cmake -S cmake -B build -G Ninja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS="-DFMT_USE_CONSTEXPR=1" \
  -D CMAKE_CUDA_FLAGS="-DFMT_USE_CONSTEXPR=1" \
  -D CMAKE_CXX_COMPILER=$PWD/lib/kokkos/bin/nvcc_wrapper \
  -D CMAKE_INSTALL_PREFIX=$SW/install \
  -D BUILD_MPI=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D PKG_KOKKOS=ON -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_${KOKKOS_ARCH}=ON \
  -D CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} \
  -D PKG_ML-IAP=ON -D PKG_ML-SNAP=ON -D MLIAP_ENABLE_PYTHON=ON -D PKG_PYTHON=ON \
  -D PKG_EXTRA-FIX=ON -D PKG_EXTRA-COMPUTE=ON -D PKG_EXTRA-DUMP=ON \
  -D PKG_MOLECULE=ON -D PKG_RIGID=ON \
  -D Python_EXECUTABLE=$(which python)

cmake --build build -j $NJ
cmake --build build --target install-python   # installs the `lammps` python module into $ENV
cd build
ls -la lmp liblammps.so*
python -c "import lammps; l=lammps.lammps(cmdargs=['-log','none','-screen','none']); \
print('LAMMPS', l.version(), '| ML-IAP', l.has_package('ML-IAP'), '| KOKKOS', l.has_package('KOKKOS'), '| PYTHON', l.has_package('PYTHON'))"
echo "BUILD OK: $SW/lammps/build/lmp   $(date)"
