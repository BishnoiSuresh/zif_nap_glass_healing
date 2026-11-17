#!/bin/bash
# =============================================================================
# STEP 3  export MACE-MH-1 (matpes_r2scan head) to the LAMMPS ML-IAP format
#   bash install/03_export_model.sh
# Output: $SW/models/mh1_matpes_r2scan_fp64-mliap_lammps.pt  (= $MODEL in config/site.sh)
# The ML-IAP export converts the model to cuEquivariance automatically.
#
# MH1_SRC: the MACE-MH-1 foundation model file. mace_mp(model="mh-1") downloads it to
# ~/.cache/mace/ on first use; set MH1_SRC if it is somewhere else.
# =============================================================================
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/site.sh
set +u; load_modules; set -u
OUT=$SW/models
MH1_SRC=${MH1_SRC:-$HOME/.cache/mace/macemh1model}
if [ ! -f "$MH1_SRC" ]; then
  python -c "from mace.calculators import mace_mp; mace_mp(model='mh-1', default_dtype='float64', head='matpes_r2scan')" || true
fi
[ -f "$MH1_SRC" ] || { echo "MACE-MH-1 model not found at $MH1_SRC -- set MH1_SRC"; exit 1; }
mkdir -p $OUT

export_one () {   # $1 src model, $2 head, $3 dtype, $4 tag
  local tmp=$OUT/$4.model
  cp "$1" $tmp
  mace_create_lammps_model $tmp --head "$2" --dtype $3 --format mliap
  mv $tmp-mliap_lammps.pt $OUT/$4-mliap_lammps.pt && rm -f $tmp
  echo "  -> $OUT/$4-mliap_lammps.pt"
}

export_one $MH1_SRC matpes_r2scan float64 mh1_matpes_r2scan_fp64    # used for all production runs
[ "${WITH_FP32:-0}" = 1 ] && export_one $MH1_SRC matpes_r2scan float32 mh1_matpes_r2scan_fp32
ls -la $OUT
