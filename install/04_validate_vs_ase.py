#!/usr/bin/env python3
"""
03_validate_vs_ase.py
=====================
Check that the LAMMPS ML-IAP model gives the SAME energy / forces / stress as the
ASE MACECalculator, then (optionally) write a
replicated data file for the speed benchmark.

For every structure:
  * ASE  : MACECalculator(model, head, float64)          -> E, F, stress
  * LAMMPS (subprocess, KOKKOS GPU, `run 0`)             -> E, F, pressure
  * prints dE/atom (meV), force MAE / max (meV/A), dP (GPa)

Pass criteria (fp64 export): |dE| < 0.1 meV/atom, force MAE < 1 meV/A.
fp32 export: expect ~0.1-1 meV/atom and a few meV/A -> fine for MD.

Usage (inside the mace-lammps env, on a GPU node):
  python 03_validate_vs_ase.py \
     --model_ase  ~/.cache/mace/macemh1model --head matpes_r2scan \
     --model_lmp  ~/software/lammps-mliap/models/mh1_matpes_r2scan_fp64-mliap_lammps.pt \
     --lmp ~/software/lammps-mliap/lammps/build/lmp \
     --structures zif62_glass_final.xyz 40nap_glass_final.xyz composite_glass_final.xyz \
     --write_bench composite_glass_final.xyz --replicate 2 2 1
"""
import argparse
import os
import re
import subprocess

import numpy as np
from ase.io import read, write

SPEC_ALL = ["C", "H", "N", "Na", "O", "P", "Zn"]


def species_of(atoms):
    present = set(atoms.get_chemical_symbols())
    return [s for s in SPEC_ALL if s in present] + sorted(present - set(SPEC_ALL))


def lammps_single_point(atoms, lmp, model, workdir, kokkos=True, d3=None):
    os.makedirs(workdir, exist_ok=True)
    spec = species_of(atoms)
    write(f"{workdir}/conf.data", atoms, format="lammps-data", specorder=spec,
          atom_style="atomic", masses=True)
    pair = f"mliap unified {model} 0"
    coeff = f"pair_coeff * * {' '.join(spec)}"
    if d3:
        pair = f"hybrid/overlay {pair} d3 9000 1600 damp_bj {d3}"
        coeff = f"pair_coeff * * mliap {' '.join(spec)}\npair_coeff * * d3 {' '.join(spec)}"
    with open(f"{workdir}/in.sp", "w") as f:
        f.write(f"""units metal
atom_style atomic
atom_modify map yes
newton on
boundary p p p
read_data conf.data
pair_style {pair}
{coeff}
neigh_modify one 3000
thermo_style custom step pe press pxx pyy pzz pxy pxz pyz
thermo_modify format float %20.10f
run 0
write_dump all custom forces.dump id fx fy fz modify sort id format float %20.12f
""")
    cmd = [lmp, "-in", "in.sp", "-log", "log.sp"]
    if kokkos:
        cmd += ["-k", "on", "g", "1", "-sf", "kk", "-pk", "kokkos", "newton", "on", "neigh", "half"]
    r = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise RuntimeError("LAMMPS failed")
    log = open(f"{workdir}/log.sp").read()
    m = re.search(r"Step\s+PotEng.*\n\s*0\s+(.+)\n", log)
    vals = [float(x) for x in m.group(1).split()]
    pe, press = vals[0], vals[1]
    lines = open(f"{workdir}/forces.dump").read().splitlines()
    i0 = [k for k, l in enumerate(lines) if l.startswith("ITEM: ATOMS")][0] + 1
    F = np.array([[float(x) for x in l.split()[1:4]] for l in lines[i0:i0 + len(atoms)]])
    return pe, F, press * 1e-4          # bar -> GPa


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_ase", required=True)
    p.add_argument("--head", default=None)
    p.add_argument("--model_lmp", required=True)
    p.add_argument("--lmp", required=True)
    p.add_argument("--structures", nargs="+", required=True)
    p.add_argument("--d3", default=None, help="e.g. r2scan -> also run MACE+D3 in LAMMPS (no ASE ref)")
    p.add_argument("--write_bench", default=None, help="structure to replicate for the speed benchmark")
    p.add_argument("--replicate", nargs=3, type=int, default=[2, 2, 1])
    p.add_argument("--workdir", default="validate_runs")
    a = p.parse_args()

    from mace.calculators import MACECalculator
    kw = dict(model_paths=a.model_ase, device="cuda", default_dtype="float64")
    if a.head:
        kw["head"] = a.head
    calc = MACECalculator(**kw)

    print(f"{'structure':38s} {'N':>5s} {'dE meV/at':>10s} {'F MAE':>8s} {'F max':>8s} {'dP GPa':>8s}")
    for s in a.structures:
        at = read(s)
        # rotate to LAMMPS' lower-triangular frame so ASE and LAMMPS forces share axes
        rcell, _ = at.cell.standard_form()
        at.set_cell(rcell, scale_atoms=True)
        at.calc = calc
        e_ase = at.get_potential_energy()
        f_ase = at.get_forces()
        p_ase = -np.mean(at.get_stress(voigt=True)[:3]) * 160.21766208   # eV/A^3 -> GPa
        tag = os.path.splitext(os.path.basename(s))[0]
        e_l, f_l, p_l = lammps_single_point(at, a.lmp, a.model_lmp, f"{a.workdir}/{tag}")
        dF = np.abs(f_l - f_ase) * 1000
        print(f"{tag:38s} {len(at):5d} {1000 * (e_l - e_ase) / len(at):10.4f} "
              f"{dF.mean():8.3f} {dF.max():8.3f} {p_l - p_ase:8.4f}")
        if a.d3:
            e_d, f_d, p_d = lammps_single_point(at, a.lmp, a.model_lmp, f"{a.workdir}/{tag}_d3", d3=a.d3)
            print(f"   + D3({a.d3}): dE_D3 = {1000 * (e_d - e_l) / len(at):.2f} meV/atom, "
                  f"dP_D3 = {p_d - p_l:.3f} GPa")

    if a.write_bench:
        at = read(a.write_bench).repeat(a.replicate)
        out = f"bench_{len(at)}atoms.data"
        write(out, at, format="lammps-data", specorder=species_of(at), atom_style="atomic", masses=True)
        print(f"\nbenchmark data: {out}  ({len(at)} atoms, species {' '.join(species_of(at))})")


if __name__ == "__main__":
    main()
