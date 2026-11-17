#!/usr/bin/env python3
"""
zif_melt_check.py -- melt criterion for ZIF (and ZIF-containing composites),
called from LAMMPS (in.melt_zif) after every hold block via `shell`.

Instead of "every element diffuses >= 6 A^2 / 10 ps" (unreachable for a ZIF
liquid below linker decomposition), a ZIF is called molten when its Zn-N
network has rearranged (Gaillac et al. Nat. Mater. 2017; Du et al. NSR 2024):

  fexch   fraction of the ORIGINAL (crystal) Zn-N bonds that are now broken
  znnew   fraction of Zn that now has >= 1 N partner it did not have at the start
  zncn4   fraction of four-coordinated Zn (info; ~0.6-0.9 in AIMD/DPMD liquids)
  msdZn   MSD of Zn vs the reference (unwrapped coords, A^2)
  msdLk   MSD of linker atoms C+N+H (A^2)
  im/bim  intact Im (C3H3N2) / bIm (C7H5N2) linkers;  broken = other C/N/H fragments
  lost    linkers lost vs the reference (decomposition guard)

Writes LAMMPS 'variable ... equal ...' lines to --out (read back with `include`).

  python zif_melt_check.py --ref zref.dump --cur zcur.dump --elements C H N Zn --out zcheck.lmp
Dumps: `write_dump all custom FILE id type xu yu zu modify sort id`.
"""
import argparse
from collections import Counter

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# hot-liquid cutoffs (slightly longer than the 300 K ones in analyze_glass.py,
# so thermal stretching at 1500 K is not counted as a broken linker)
CUT_LINK = {("C", "C"): 1.85, ("C", "N"): 1.85, ("N", "N"): 1.70, ("C", "H"): 1.40, ("N", "H"): 1.35}
ZN_N = 2.5


def read_dump(fn):
    with open(fn) as f:
        lines = f.read().splitlines()
    k = next(i for i, l in enumerate(lines) if l.startswith("ITEM: NUMBER OF ATOMS"))
    n = int(lines[k + 1])
    kb = next(i for i, l in enumerate(lines) if l.startswith("ITEM: BOX BOUNDS"))
    b = np.array([[float(x) for x in lines[kb + 1 + d].split()] for d in range(3)])
    tilt = b.shape[1] == 3
    xy, xz, yz = (b[0, 2], b[1, 2], b[2, 2]) if tilt else (0.0, 0.0, 0.0)
    xlo = b[0, 0] - min(0.0, xy, xz, xy + xz); xhi = b[0, 1] - max(0.0, xy, xz, xy + xz)
    ylo = b[1, 0] - min(0.0, yz);              yhi = b[1, 1] - max(0.0, yz)
    zlo, zhi = b[2, 0], b[2, 1]
    cell = np.array([[xhi - xlo, 0, 0], [xy, yhi - ylo, 0], [xz, yz, zhi - zlo]])
    ka = next(i for i, l in enumerate(lines) if l.startswith("ITEM: ATOMS"))
    cols = lines[ka].split()[2:]
    d = np.array([l.split() for l in lines[ka + 1:ka + 1 + n]], dtype=float)
    d = d[np.argsort(d[:, cols.index("id")])]
    pc = [c for c in ("xu", "x") if c in cols][0]
    i0 = cols.index(pc)
    return d[:, cols.index("type")].astype(int), d[:, i0:i0 + 3], cell, np.array([xlo, ylo, zlo])


def atoms_of(types, pos, cell, origin, elements):
    sym = [elements[t - 1] for t in types]
    at = Atoms(sym, positions=pos - origin, cell=cell, pbc=True)
    at.wrap()
    return at


def znn_pairs(at, s):
    if not (s == "Zn").any():
        return set(), np.array([])
    i, j = neighbor_list("ij", at, {("Zn", "N"): ZN_N})
    m = s[i] == "Zn"
    pairs = set(zip(i[m].tolist(), j[m].tolist()))
    cn = np.bincount(i[m], minlength=len(at))[s == "Zn"]
    return pairs, cn


def linkers(at, s):
    idx = np.where(np.isin(s, ["C", "N", "H"]))[0]
    if len(idx) == 0:
        return 0, 0, 0
    sub = at[idx]
    i, j = neighbor_list("ij", sub, CUT_LINK)
    _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(sub),) * 2))
    ss = s[idx]
    c = Counter("".join(f"{e}{n}" for e, n in sorted(Counter(ss[lab == k]).items())) for k in np.unique(lab))
    im, bim = c.pop("C3H3N2", 0), c.pop("C7H5N2", 0)
    return im, bim, sum(c.values())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True)
    p.add_argument("--cur", required=True)
    p.add_argument("--elements", nargs="+", required=True)
    p.add_argument("--out", default="zcheck.lmp")
    a = p.parse_args()

    tr, xr, cr, orr = read_dump(a.ref)
    tc, xc, cc, oc = read_dump(a.cur)
    ref, cur = atoms_of(tr, xr, cr, orr, a.elements), atoms_of(tc, xc, cc, oc, a.elements)
    s = np.array(cur.get_chemical_symbols())

    pr, _ = znn_pairs(ref, s)
    pc, cn = znn_pairs(cur, s)
    fexch = 1 - len(pr & pc) / max(len(pr), 1)
    zn = np.where(s == "Zn")[0]
    znnew = np.mean([any((z, n) not in pr for (zz, n) in pc if zz == z) for z in zn]) if len(zn) else 0.0
    zncn4 = float(np.mean(cn == 4)) if len(zn) else 0.0

    # MSD from unwrapped coordinates, centre-of-mass drift removed
    dx = xc - xr
    dx -= dx.mean(axis=0)
    d2 = (dx ** 2).sum(1)
    msdZn = float(d2[s == "Zn"].mean()) if len(zn) else 0.0
    lk = np.isin(s, ["C", "N", "H"])
    msdLk = float(d2[lk].mean()) if lk.any() else 0.0

    im0, bim0, _ = linkers(ref, s)
    im, bim, broken = linkers(cur, s)
    lost = (im0 + bim0) - (im + bim)

    v = dict(zfexch=fexch, zznnew=znnew, zzncn4=zncn4, zmsdZn=msdZn, zmsdLk=msdLk,
             zim=im, zbim=bim, zbroken=broken, zlost=lost)
    with open(a.out, "w") as f:
        for k, x in v.items():
            f.write(f"variable {k} equal {x:.6g}\n")
    print("ZCHECK " + " ".join(f"{k}={x:.4g}" for k, x in v.items()))


if __name__ == "__main__":
    main()
