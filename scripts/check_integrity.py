#!/usr/bin/env python3
"""
check_integrity.py -- PASS/FAIL structure check after every build / crack / heal .
Catches e.g. linkers fused or split by a crack/build step before any analysis is done.

  python check_integrity.py healed.data --elements C H N Na O P Zn [--zn_expected 64]
Checks: all CNH units are Im (C3H3N2) or bIm (C7H5N2) and their numbers match Zn
(Im = 1.75 Zn, bIm = 0.25 Zn); no fused/broken fragments; P four-coordinated (>= 97 %);
no O2 molecules beyond --o2_max; no heavy-atom contact < 1.2 A.
Exit status 0 = PASS, 1 = FAIL (the job script prints a warning and keeps going).
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_composite import integrity_report  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("--elements", nargs="+", required=True)
ap.add_argument("--zn_expected", type=int, default=None)
ap.add_argument("--o2_max", type=int, default=3, help="the NaP glasses carry 0-2 O2 from the random start")
ap.add_argument("--p4_min", type=float, default=0.97)
a = ap.parse_args()

Z = {k + 1: atomic_numbers[e] for k, e in enumerate(a.elements)}
at = read(a.data, format="lammps-data", Z_of_type=Z, atom_style="atomic")
s = np.array(at.get_chemical_symbols())
rep = integrity_report(at, os.path.basename(a.data))
fail = []
nzn = int((s == "Zn").sum())
if a.zn_expected is not None and nzn != a.zn_expected:
    fail.append(f"Zn count {nzn} != {a.zn_expected}")
if nzn:
    im_exp, bim_exp = round(1.75 * nzn), round(0.25 * nzn)
    if rep.get("Im") != im_exp or rep.get("bIm") != bim_exp:
        fail.append(f"linkers Im {rep.get('Im')}/{im_exp}, bIm {rep.get('bIm')}/{bim_exp}")
    if rep.get("broken_CNH_fragments", 0):
        fail.append(f"{rep['broken_CNH_fragments']} broken/fused CNH fragments {rep.get('fragment_types')}")
if "P" in s:
    cn = rep.get("P_CN(O)", {})
    tot = sum(cn.values())
    if tot and cn.get(4, 0) / tot < a.p4_min:
        fail.append(f"only {cn.get(4, 0)}/{tot} P four-coordinated")
    if rep.get("O-O<1.6A (O2)", 0) > a.o2_max:
        fail.append(f"{rep['O-O<1.6A (O2)']} O2 molecules")
heavy = np.where(s != "H")[0]
i, j, d = neighbor_list("ijd", at[heavy], 1.2)
if len(i):
    sh = s[heavy]
    fail.append(f"{len(i) // 2} heavy-atom contacts < 1.2 A: {Counter('-'.join(sorted((sh[p], sh[q]))) for p, q in zip(i, j) if p < q)}")
print(f"INTEGRITY {'PASS' if not fail else 'FAIL'} {a.data} ({len(at)} atoms, {nzn} Zn)" + ("" if not fail else ": " + "; ".join(fail)))
sys.exit(0 if not fail else 1)
