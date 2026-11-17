# Simulation data

Final structures, two representative trajectories, thermodynamic records, elastic tensors
and result tables of the runs behind the paper (ZIF-62/40NaP composite; interfacial crack
healed at 500, 600, 700 and 800 K, 3 seeds each). Full trajectories (several GB) are not
archived; every file here can be regenerated with the code in this repository (top-level
README), and the tables can be recomputed from the structures with `post/`.

## structures/  (LAMMPS data files, `atom_style atomic`, Å)

Atom-type order: 40NaP `Na O P`; ZIF-62 `C H N Zn`; composite and everything derived from it
`C H N Na O P Zn`. Read with ASE as in `post/common.py` (atom ids are kept, so ASE's id order
matches `crack_side.txt`). In all composite files the ZIF layer is centred along z (circular mean
of Zn at L_z/2), so the cracked interface I1 lies below the ZIF and the intact I2 above it;
these files open directly in OVITO.

| file | content |
|---|---|
| `input/zif62_ordered_unitcell.cif` | **input**: ordered (P1) unit cell of crystalline ZIF-62, Zn(Im)₁.₇₅(bIm)₀.₂₅, 296 atoms (16 Zn, 28 Im, 4 bIm), derived from CSD deposition 2130183 (doi:10.5517/ccdc.csd.cc29hml1) by resolving its disorder into one ordered arrangement. Both supercells below are exact replicas of this cell |
| `input/zif62_crystal_2x2x1.data` | **input**: 2×2×1 supercell, 1184 atoms (`C H N Zn`); start of `STEP=zif` (the ZIF layer of the composite) |
| `input/zif62_crystal_2x2x2.data` | **input**: 2×2×2 supercell, 2368 atoms; start of `STEP=zif CELL=222` (the ZIF-62 T_g run) |

Everything outside `input/` is simulation output.

| file | content |
|---|---|
| `zif62_glass_300K.data` | ZIF-62 glass layer, 2×2×1 cell, 1184 atoms (112 Im + 16 bIm) |
| `nap_layer_glass_300K.data` | 40NaP glass, 2160 atoms, lateral box of the ZIF layer (used in the composite) |
| `nap_iso_1GPa_glass_300K.data` | free-box 40NaP, no D3, 1 GPa quench (X-ray comparison) |
| `nap_iso_D3_glass_300K.data` | free-box 40NaP, MACE + D3 at 0 bar (T_g run) |
| `composite_start.data` | NaP│ZIF│NaP stack as built by `build_composite.py`, 3344 atoms: the starting configuration of the anneal |
| `composite_pristine_300K.data` | annealed composite glass ("pristine") |
| `cracked_t0.data` | 5 Å crack at interface I1, as cut (15 Zn–O–P bonds broken); identical for every healing run |
| `crack_side.txt`, `crack_meta.json` | side label per atom and crack-plane metadata from `make_crack.py`. `centred_files` in `crack_meta.json` gives, for each file here, the crack-plane position and the second side-label boundary in the centred cell: `crack_plane(at, side, plane_frac, 2, origin=origin_frac)` (`scripts/heal_analysis_r1.py`) reproduces the across-plane bond counts of the tables |
| `contact_5ps_300K.data` | cracked cell after 5 ps at 300 K with the box height fixed (surfaces in contact, before heating; from the 800 K seed-1 run) |
| `healed_T<T>_s<seed>.data` | healed structures at 300 K, T = 500/600/700/800 K, seeds 1–3 |

## trajectories/

- `composite_anneal.xyz.gz`: the composite anneal MD, 63 frames every 5 ps (extended XYZ;
  300 → 1200 K, 40 ps at 1200 K, quench at 5 K/ps, release and anneal at 300 K). Per-frame
  `time_ps`, `T_K`, `N_ZnOP`; per-atom flag `ZnOP` = 1 for Zn and O atoms in Zn–O–P bonds.
- `heal_T800_s1_ZIFcentred.lammpstrj.gz`: the 800 K healing run (seed 1), frames every 5 ps,
  ZIF layer centred along z (LAMMPS text dump: id type element x y z; time step 1 fs).

## curves/  (LAMMPS `fix ave/time`, 1 ps averages)

- `*_cooling_curve.dat`, `composite_anneal_curve.dat`: step, T (K), P (bar), PE (eV), V (Å³), density (g/cm³)
- `heal_curve_T<T>_s<seed>.dat`: step, T, P, PE, density, Lx, Ly, Lz (time step 1 fs: step 1000 = 1 ps)

## elastic/

`elastic_C_<structure>.txt` for `pristine`, `cracked`, `contact` and `healed_T<T>_s<seed>`: 6×6 stiffness
matrix (GPa) from ±0.5 % strains of the minimised cell (`lammps/in.elastic`); rows = stress
components xx yy zz yz xz xy, columns = applied strain. The cracked cell carries no load
across the open gap, so its tensor is unstable (C33 < 0).

## tables/

| file | content |
|---|---|
| `Table_interface_metrics_all_structures.csv` | per structure: Zn–O–P, Na–O, P–O bonds across the crack plane; Zn–O–P, P_ZnOP, ICC, IBD at I1 and I2; density, L_z, area, minimum density at I1 |
| `Table_IBD_ICC_per_area.csv` | per structure plus mean/sd per T: Zn–O–P bonds, IBD (per nm²), Zn atoms, connected Zn, ICC, connected Zn per nm² and P_ZnOP, for the crack plane, I1, I2 and total |
| `Table_healing_summary_by_T.csv` | healed state per T: mean and sd over the 3 seeds |
| `Table_elastic_all_structures.csv` | C_ij, E_x, E_y, E_z, Voigt–Reuss–Hill B, G, E and the smallest eigenvalue of C per structure (cracked: E_z = 0, B/G/E not defined) |
| `Table_elastic_summary_by_T.csv` | healed moduli per T: mean, sd and % of pristine |

Definitions (top-level README): Zn–O ≤ 2.5 Å, Zn–N ≤ 2.5 Å, P–O ≤ 2.05 Å; Zn–O–P = Zn–O bond
whose O is bonded to P; IBD = Zn–O–P per nm² of interface; P_ZnOP = Zn–O–P / (Zn–N + Zn–O);
ICC = % of Zn with ≥ 1 Zn–O–P; I1 = cracked interface, I2 = intact interface.
