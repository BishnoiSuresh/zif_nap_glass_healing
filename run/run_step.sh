#!/bin/bash
# =============================================================================
# run_step.sh -- one step of the ZIF-62 / 40NaP hybrid-glass healing workflow.
# Scheduler-independent: run it directly on a GPU node, or call it from a batch
# script (see run/pbs_job.pbs and run/slurm_job.sh). Each step needs 1 GPU.
#
#   cd <RUN directory>            # every step writes its own sub-folder here
#   STEP=... [POT=...] [...] bash <repo>/run/run_step.sh
#
#  POT  B = MACE-MH-1 (matpes_r2scan), no D3; glasses quenched at 1 GPa and released at
#           300 K (compensates the missing dispersion). Used for everything in the paper's
#           composite / crack / healing results.   (default)
#       A = MACE-MH-1 + D3(BJ), 0 bar. Used only for the free-box 40NaP Tg (STEP=nap MODE=iso).
#
#  STEP=zif  [CELL=221|222]       ZIF-62 crystal -> glass. 221 (default): 2x2x1, 1184 at.,
#                                 the composite layer; 222: 2x2x2, 2368 at., the T_g run -> zif<CELL>_B/
#  STEP=nap   MODE=iso|layer      40NaP 2160 at., 5 K/ps; iso = free box (Tg, XRD),
#                                 layer = lateral box of the ZIF layer (BARO z)      -> nap_<MODE>_<POT>/
#  STEP=composite                 NaP|ZIF|NaP stack (3344 at., 3.24:1 wt), anneal 1200 K,
#                                 quench 5 K/ps, integrity + interface metrics       -> composite_B/
#  STEP=heal  THEAL=<K> SEED=1|2|3 [THOLD=100] [PHEAL=1000] [SUF=]
#                                 interfacial crack (interface I1, 5 A) + healing    -> heal_B_T<THEAL>_s<SEED><SUF>/
#  STEP=elastic ITEMS=a+b+...     0 K elastic tensors; items: pristine | cracked | contact |
#                                 heal_B_T<T>_s<seed> (its healed structure)          -> elastic_B/
#  STEP=tension ITEMS=a+b+...     optional uniaxial tension along z, same items       -> tension_B/
#
# Order: zif -> nap (layer) -> composite -> heal (x T x seeds) -> elastic.
# The exact list of runs used for the paper is in run/paper_runs.sh.
# Existing outputs are never overwritten unless FORCE=1.
# =============================================================================
set -eo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
source $REPO/config/site.sh
RUN=${RUN:-${PBS_O_WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}}
P=$REPO/lammps; S=$REPO/scripts
POT=${POT:-B}
case $POT in
  A) D3=1; PQ=0;     PQH=0 ;;
  B) D3=0; PQ=10000; PQH=1 ;;
  *) echo "POT must be A or B"; exit 2 ;;
esac
ZREF=$RUN/zif221_B/glass_300K.dat; ZREP="1 1"          # ZIF-62 glass layer (2x2x1, 1184 atoms)
EL="C H N Na O P Zn"
NGPU_ARGS=${NGPU_ARGS:-"-k on g 1 -sf kk -pk kokkos newton on neigh half"}
MPIRUN=${MPIRUN:-"mpirun -np 1"}

echo "node $(hostname)  job ${PBS_JOBID:-${SLURM_JOB_ID:-}}  $(date)"
echo "STEP=$STEP POT=$POT (D3=$D3, quench pressure $PQ bar, PQHOLD=$PQH)  RUN=$RUN  REPO=$REPO"
load_modules
lmp () { $MPIRUN $LMP $NGPU_ARGS -var MODEL $MODEL -var D3 $D3 "$@"; }
fresh () { if [ -f "$1/log.lammps" ] && [ "${FORCE:-0}" != 1 ]; then echo "ERROR: $1 already has a log.lammps (set FORCE=1 to overwrite)"; exit 1; fi; mkdir -p "$1"; }
need () { [ -f "$1" ] || { echo "ERROR: missing $1 -- run the step that makes it first"; exit 2; }; }
check () { python $S/check_integrity.py "$@" || echo "WARNING: integrity check FAILED -- inspect before using this structure"; }
onlyB () { [ $POT = B ] || { echo "STEP=$STEP is only set up for POT=B"; exit 2; }; }

case $STEP in
# -----------------------------------------------------------------------------
zif)      # ZIF-62 2x2x1 crystal -> glass: network melt at 1500 K, cool to 1300 K while P -> 1 GPa,
          # 5 K/ps quench at 1 GPa, release to 0 bar at 300 K
  onlyB
  CELL=${CELL:-221}
  case $CELL in
    221) ZDATA=$REPO/data/structures/input/zif62_crystal_2x2x1.data; ZSEED=5928456; LOST=2; NZN=64 ;;
    222) ZDATA=$REPO/data/structures/input/zif62_crystal_2x2x2.data; ZSEED=4929456; LOST=1; NZN=128 ;;   # T_g run
    *) echo "CELL must be 221 or 222"; exit 2 ;;
  esac
  W=$RUN/zif${CELL}_B; fresh $W; cd $W; need $ZDATA
  lmp -in $P/in.melt_zif -log log.lammps -var DATAFILE $ZDATA -var ELEMENTS "C H N Zn" \
    -var SEED $ZSEED -var DT 0.001 -var BARO iso -var RATE 5 -var THOT 1500 -var PEND 10000 -var PQHOLD 1 \
    -var LOSTMAX $LOST -var ZCHECK $S/zif_melt_check.py > lammps.out 2> lammps.err
  grep -E "### MELT|Liquid at TREL|quenched at|RESULT" log.lammps | tail -6
  check glass_300K.dat --elements C H N Zn --zn_expected $NZN
  python $S/analyze_glass.py glass_300K.dat --elements C H N Zn ;;
# -----------------------------------------------------------------------------
nap)      # 40NaP 2160 atoms at 5 K/ps (MODE=iso: free box; MODE=layer: lateral = ZIF layer)
  MODE=${MODE:-layer}
  W=$RUN/nap_${MODE}_${POT}; fresh $W; cd $W
  if [ $MODE = layer ]; then
    onlyB; need $ZREF
    read LX LY < <(python - "$ZREF" $ZREP <<'PY'
import sys
from ase.io import read
a = read(sys.argv[1], format="lammps-data", Z_of_type={1: 6, 2: 1, 3: 7, 4: 30}, atom_style="atomic")
c = a.cell.cellpar(); print(int(sys.argv[2]) * c[0], int(sys.argv[3]) * c[1])
PY
)
    echo "NaP layer lateral box $LX x $LY A (ZIF $ZREF)"
    python $S/prepare_inputs.py nap --atoms 2160 --lateral $LX $LY --seed ${NSEED:-21} --out nap_start.data
    BARO=z; BRELAX=z
  else
    python $S/prepare_inputs.py nap --atoms 2160 --seed ${NSEED:-21} --out nap_start.data
    BARO=iso; BRELAX=aniso
  fi
  lmp -in $P/in.quench_zifnap -log log.lammps -var DATAFILE nap_start.data -var ELEMENTS "Na O P" \
      -var SEED $((4939456 + ${NSEED:-21})) -var DT 0.002 -var BARO $BARO -var BRELAX $BRELAX \
      -var T0 2000 -var TMAX 2600 -var TMIX 50 -var MIXMAX 2 -var THOLD 20 -var RATE 5 \
      -var TANNEAL 20 -var TSAMPLE 10 -var FLOGN 1000 -var PEND $PQ -var PQHOLD $PQH > lammps.out 2> lammps.err
  grep -E "THOT =|Liquid at|quenched at|RESULT" log.lammps | tail -5
  check glass_300K.dat --elements Na O P
  python $S/analyze_glass.py glass_300K.dat --elements Na O P ;;
# -----------------------------------------------------------------------------
composite)  # ZIF 1184 at. + NaP 2160 at. (3.24:1 wt), anneal 1200 K 40 ps, quench 5 K/ps at 1 GPa
  onlyB
  NAP=$RUN/nap_layer_B/glass_300K.dat; need $ZREF; need $NAP
  W=$RUN/composite_B; fresh $W; cd $W
  python $S/build_composite.py --zif $ZREF --nap $NAP --zif_rep $ZREP --n_zif 1 --n_nap 1 --seed 1 \
         --output composite_start.xyz | tee build.log
  check composite_start.data --elements $EL --zn_expected 64
  lmp -in $P/in.composite_anneal -log log.lammps -var DATAFILE composite_start.data -var TANN ${TANN:-1200} \
      -var THOLD 40 -var HEATRATE 20 -var RATE 5 -var TANNEAL 20 -var TSAMPLE 10 \
      -var PANN $PQ -var PQHOLD $PQH > lammps.out 2> lammps.err
  grep RESULT log.lammps
  check composite_glass.data --elements $EL --zn_expected 64
  python $S/heal_analysis_r1.py --elements $EL --states pristine:composite_glass.data --out pristine ;;
# -----------------------------------------------------------------------------
heal)     # crack at interface I1 (5 A) + heal at THEAL with PHEAL bar normal to the crack
  onlyB
  [ -n "$THEAL" ] || { echo "set THEAL"; exit 2; }
  SEED=${SEED:-1}
  C0=$RUN/composite_B/composite_glass.data; need $C0
  W=$RUN/heal_B_T${THEAL}_s${SEED}${SUF:-}; fresh $W; cd $W   # SUF e.g. _p0 for a PHEAL=0 variant
  python $S/make_crack.py $C0 --elements $EL --axis z --plane auto --interface 1 --gap ${GAP:-5} --out cracked.data
  check cracked.data --elements $EL --zn_expected 64
  lmp -in $P/in.heal_r1 -log log.lammps -var DATAFILE cracked.data -var TAG healed -var CAX z \
      -var THEAL $THEAL -var PHEAL ${PHEAL:-1000} -var THOLD ${THOLD:-100} -var HEATRATE 20 \
      -var RATE 20 -var SEED $((7770 + SEED)) > lammps.out 2> lammps.err
  grep RESULT log.lammps
  check healed.data --elements $EL --zn_expected 64
  python $S/heal_analysis_r1.py --elements $EL --states pristine:$C0 cracked:cracked.data \
         contact:start_300K_healed.data healed:healed.data --crack . --traj heal_avg_healed.lammpstrj --out heal
  python $S/center_dump.py heal_healed.lammpstrj heal_healed_centered.lammpstrj || true ;;
# -----------------------------------------------------------------------------
elastic|tension)
  onlyB
  [ -n "$ITEMS" ] || { echo "set ITEMS, e.g. ITEMS=pristine+cracked+contact+heal_B_T800_s1"; exit 2; }
  ITEMS=$(echo "$ITEMS" | tr '+' ' ')
  W=$RUN/${STEP}_B; mkdir -p $W; cd $W
  for it in $ITEMS; do
    case $it in
      pristine) D=$RUN/composite_B/composite_glass.data; T=pristine; R="aniso 0.0" ;;
      cracked)  # the crack is deterministic (same cut for every heal run): take it from any heal folder
                HD=$(ls -d $RUN/heal_B_T*_s* 2>/dev/null | head -1); D=$HD/cracked.data; T=cracked
                R="x 0.0 y 0.0" ;;                         # open crack as cut (t = 0), z not relaxed
      contact)  # after 5 ps at 300 K with L_z fixed (velocity-seed dependent; the paper uses heal_B_T800_s1)
                HD=$(ls -d $RUN/heal_B_T800_s1 $RUN/heal_B_T*_s* 2>/dev/null | head -1); D=$HD/start_300K_healed.data; T=contact
                R="x 0.0 y 0.0" ;;
      heal_B_T*_s*) D=$RUN/$it/healed.data; T=healed_${it#heal_B_}; R="aniso 0.0" ;;
      *) echo "unknown item $it (use pristine, cracked, contact or heal_B_T<T>_s<seed>)"; exit 2 ;;
    esac
    need $D
    if [ $STEP = elastic ]; then
      [ -f elastic_C_$T.txt ] && [ "${FORCE:-0}" != 1 ] && { echo "skip $T (done)"; continue; }
      lmp -in $P/in.elastic -log log_$T.lammps -var DATAFILE $D -var TAG $T -var RELAX "$R" > out_$T.txt 2> err_$T.txt
      python $S/elastic_summary.py elastic_C_$T.txt
    else
      [ -f stress_strain_$T.dat ] && [ "${FORCE:-0}" != 1 ] && { echo "skip $T (done)"; continue; }
      lmp -in $P/in.tension -log log_$T.lammps -var DATAFILE $D -var TAG $T -var ERATE ${ERATE:-0.001} \
          -var EMAX ${EMAX:-0.15} > out_$T.txt 2> err_$T.txt
      grep RESULT log_$T.lammps
    fi
  done ;;
*) echo "unknown STEP=$STEP"; exit 2 ;;
esac
echo "done $(date)"
