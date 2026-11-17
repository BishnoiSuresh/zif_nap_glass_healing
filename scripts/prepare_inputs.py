#!/usr/bin/env python3
"""
prepare_inputs.py  --  LAMMPS data files (random 40NaP liquid start, ZIF-62 crystal) for the melt-quench runs
=======================================================================
Element (type) order is fixed per system and printed, use it for -var ELEMENTS:
    40NaP      : Na O P
    ZIF-62     : C H N Zn
    composite  : C H N Na O P Zn      (written by build_composite.py)

Sub-commands
------------
  nap      random 40Na2O-60P2O5 box at target density (safe packing: PBC,
           pair-specific minimum distances, no unchecked placement)
             python prepare_inputs.py nap --atoms 540 --density 2.5 --seed 1 --out 40nap_s1.data
  convert  any ASE-readable structure (CIF/POSCAR/xyz) -> LAMMPS data
             python prepare_inputs.py convert zif62_correct.xyz --elements C H N Zn \
                    --repeat 1 1 1 --out zif62_crystal.data
"""
import argparse

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write

NAP_MIN_DIST = {("P", "P"): 2.90, ("O", "O"): 2.20, ("P", "O"): 1.45,
                ("Na", "Na"): 2.80, ("Na", "P"): 2.80, ("Na", "O"): 2.10}


def random_pack(symbols, L, min_dist, seed, max_attempts=20000, max_restarts=20,
                order=("P", "Na", "O")):
    """Random sequential packing with minimum distances (PBC, orthorhombic box L[3]).
    Cell-list version: O(N) per insertion, fine for ~10^4 atoms."""
    L = np.broadcast_to(np.asarray(L, float), (3,)).copy()
    els = sorted(set(symbols))
    dmin = {(a, b): min_dist.get((a, b), min_dist.get((b, a))) for a in els for b in els}
    rc = max(dmin.values())
    nc = np.maximum((L // rc).astype(int), 1)
    cs = L / nc
    rank = {e: i for i, e in enumerate(order)}
    idx = sorted(range(len(symbols)), key=lambda k: rank.get(symbols[k], 99))
    offs = {tuple(o) for o in np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])}
    for restart in range(max_restarts):
        rng = np.random.default_rng(seed + 1000 * restart)
        pos = np.empty((len(symbols), 3))
        cells = {}
        ok, nplaced = True, 0
        for k in idx:
            s = symbols[k]
            for _ in range(max_attempts):
                tr = rng.random(3) * L
                c = (tr // cs).astype(int) % nc
                near = [m for o in {tuple((c + np.array(o)) % nc) for o in offs} for m in cells.get(o, ())]
                if near:
                    d = pos[near] - tr
                    d -= L * np.round(d / L)
                    thr = np.array([dmin[(s, symbols[m])] for m in near])
                    if np.any(np.einsum("ij,ij->i", d, d) < thr ** 2):
                        continue
                pos[k] = tr
                cells.setdefault(tuple(c), []).append(k)
                nplaced += 1
                break
            else:
                ok = False
                break
        if ok:
            return pos
        print(f"  restart {restart + 1}: could not place atom {nplaced + 1}")
    raise RuntimeError("packing failed: lower density or min distances")


def cmd_nap(a):
    s = a.atoms / 540.0
    n_na, n_p, n_o = round(80 * s), round(120 * s), round(340 * s)
    sym = ["Na"] * n_na + ["P"] * n_p + ["O"] * n_o
    m = sum(atomic_masses[atomic_numbers[x]] for x in sym)
    V = m * 1.66054 / a.density
    if a.lateral:   # fixed lateral box (to match a ZIF layer); height from the density
        L = np.array([a.lateral[0], a.lateral[1], V / (a.lateral[0] * a.lateral[1])])
    else:
        L = np.array([V ** (1 / 3)] * 3)
    pos = random_pack(sym, L, NAP_MIN_DIST, a.seed)
    at = Atoms(sym, positions=pos, cell=L, pbc=True)
    write(a.out, at, format="lammps-data", specorder=["Na", "O", "P"], atom_style="atomic", masses=True)
    print(f"{a.out}: Na{n_na} O{n_o} P{n_p} ({len(sym)} atoms), rho={a.density}  box={np.round(L, 3)}  ELEMENTS=\"Na O P\"")


def cmd_convert(a):
    at = read(a.structure)
    if a.repeat != [1, 1, 1]:
        at = at.repeat(a.repeat)
    rcell, _ = at.cell.standard_form()
    at.set_cell(rcell, scale_atoms=True)
    at.wrap()
    present = set(at.get_chemical_symbols())
    if present - set(a.elements):
        raise SystemExit(f"elements {present - set(a.elements)} missing from --elements")
    write(a.out, at, format="lammps-data", specorder=a.elements, atom_style="atomic", masses=True)
    rho = at.get_masses().sum() * 1.66054 / at.get_volume()
    print(f"{a.out}: {len(at)} atoms {at.get_chemical_formula()} rho={rho:.3f}  ELEMENTS=\"{' '.join(a.elements)}\"")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    n = sp.add_parser("nap")
    n.add_argument("--atoms", type=int, default=540)
    n.add_argument("--density", type=float, default=2.5)
    n.add_argument("--seed", type=int, default=1)
    n.add_argument("--lateral", nargs=2, type=float, default=None, metavar=("LX", "LY"),
                   help="fixed lateral box (A); run the quench with BARO=z BRELAX=z to keep it")
    n.add_argument("--out", required=True)
    c = sp.add_parser("convert")
    c.add_argument("structure")
    c.add_argument("--elements", nargs="+", required=True)
    c.add_argument("--repeat", nargs=3, type=int, default=[1, 1, 1])
    c.add_argument("--out", required=True)
    a = p.parse_args()
    {"nap": cmd_nap, "convert": cmd_convert}[a.cmd](a)


if __name__ == "__main__":
    main()
