#!/usr/bin/env python3
"""
heal_analysis_r1.py -- interface metrics for the interfacial-crack healing runs.

For every structure (pristine, cracked, healed, ...) and for each interface separately
(I1 = the cracked one, below the ZIF centre; I2 = intact control, above):
  N_ZnOP    Zn-O bonds whose O is bonded to P              (Zn-O 2.5 A, P-O 2.05 A)
  IBD       N_ZnOP / A            (per nm2 of interface, A = lateral area)
  P_ZnOP    N_ZnOP / N(Zn-N + Zn-O) of the Zn in that half (%)   <- bond population
  ICC       % of Zn with >= 1 Zn-O-P bond (cf. 67Zn NMR: 71 % in 1:1, 89 % in 1:5)
Zn are assigned to I1 or I2 by their position relative to the ZIF centre (circular mean of Zn
along the layer normal), which works for pristine, cracked and healed cells alike.
Also: bonds across the crack plane per nm2 (needs crack_side.txt + crack_meta.json), and
IBD(t) during the heal from the time-averaged dump (heal_avg_<tag>.lammpstrj).

  python heal_analysis_r1.py --elements C H N Na O P Zn --states pristine:../composite_B/composite_glass.data \
       cracked:start_300K_healed.data healed:healed.data --crack . --traj heal_avg_healed.lammpstrj --out r1
Writes <out>_interfaces.csv, <out>_crackplane.csv, <out>_ibd_t.csv, <out>_summary.json
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list

CUT = {("Zn", "N"): 2.5, ("Zn", "O"): 2.5, ("P", "O"): 2.05, ("Na", "O"): 3.0}


def metrics(at, ax=2):
    s = np.array(at.get_chemical_symbols())
    L = at.cell.lengths()
    A = L[0] * L[1] * L[2] / L[ax] / 100.0                 # nm2, plane normal to ax
    i, j = neighbor_list("ij", at, CUT)
    ob = np.zeros(len(at), bool)
    ob[j[(s[i] == "P") & (s[j] == "O")]] = True
    m = s[i] == "Zn"
    zi, zj = i[m], j[m]
    isN = s[zj] == "N"
    isO = s[zj] == "O"
    isZOP = isO & ob[zj]
    zn = np.where(s == "Zn")[0]
    z = at.get_positions()[:, ax] % L[ax]
    zc = np.angle(np.exp(2j * np.pi * z[zn] / L[ax]).mean()) / (2 * np.pi) * L[ax]
    u = (z - zc + L[ax] / 2) % L[ax] - L[ax] / 2
    half = np.where(u < 0, 1, 2)
    out = {}
    for lab, sel_zn in (("I1", zn[half[zn] == 1]), ("I2", zn[half[zn] == 2]), ("total", zn)):
        inz = np.isin(zi, sel_zn)
        nzop = int((isZOP & inz).sum())
        nall = int(((isN | isO) & inz).sum())
        nzn_w = len(np.unique(zi[isZOP & inz]))
        out[lab] = dict(N_ZnOP=nzop, N_ZnN=int((isN & inz).sum()), N_ZnO=int((isO & inz).sum()),
                        IBD_nm2=nzop / (A * (2 if lab == "total" else 1)),
                        P_ZnOP_pct=100.0 * nzop / max(nall, 1), Zn=len(sel_zn), Zn_with_ZnOP=nzn_w,
                        ICC_pct=100.0 * nzn_w / max(len(sel_zn), 1))
    return out, dict(area_nm2=A, zif_centre=float(zc))


def crack_plane(at, side, plane, ax=2, origin=0.0):
    """bonds across the crack plane per nm2. side = per-atom side labels from make_crack.py; they
    flip at the crack plane `plane` and at the original cell boundary `origin` (fractions of the
    cell along ax; origin = 0 for the run's own files, see crack_meta.json for shifted files)."""
    s = np.array(at.get_chemical_symbols())
    L = at.cell.lengths()
    f = at.get_scaled_positions()[:, ax]
    i, j, D = neighbor_list("ijD", at, CUT)
    mid = (f[i] + 0.5 * D[:, ax] / L[ax]) % 1
    dist = np.abs(mid - plane)
    dist = np.minimum(dist, 1 - dist)
    do = np.abs(mid - origin % 1.0)
    near = dist < np.minimum(do, 1 - do)                   # closer to the crack plane than to the original cell
                                                           # boundary (the other place where the side label flips)
    cr = (side[i] != side[j]) & near & (i < j)
    ob = np.zeros(len(at), bool)
    pi_, oj = neighbor_list("ij", at, {("P", "O"): 2.05})
    ob[oj[s[pi_] == "P"]] = True
    c = Counter()
    for p, q in zip(i[cr], j[cr]):
        k = "-".join(sorted((s[p], s[q])))
        c[k] += 1
        if k == "O-Zn" and (ob[p] or ob[q]):
            c["Zn-O-P"] += 1
    A = L[0] * L[1] * L[2] / L[ax] / 100.0
    return {k: c.get(k, 0) / A for k in ("Zn-O-P", "O-Zn", "N-Zn", "O-P", "Na-O")}


def read_avg_dump(fn, elements):
    """frames of the time-averaged dump: id type element f_avx[1..3] (unwrapped averages)."""
    frames = []
    with open(fn) as fh:
        lines = fh.read().splitlines()
    k = 0
    while k < len(lines):
        if lines[k].startswith("ITEM: TIMESTEP"):
            step = int(lines[k + 1])
            n = int(lines[k + 3])
            b = np.array([[float(v) for v in lines[k + 5 + d].split()[:2]] for d in range(3)])
            cols = lines[k + 8].split()[2:]
            rows = np.array([l.split() for l in lines[k + 9:k + 9 + n]])
            ids = rows[:, cols.index("id")].astype(int)
            o = np.argsort(ids)
            el = rows[o, cols.index("element")] if "element" in cols else \
                np.array([elements[int(t) - 1] for t in rows[o, cols.index("type")]])
            xyz = rows[o][:, [c for c, name in enumerate(cols) if name.startswith(("f_avx", "x", "y", "z"))][:3]].astype(float)
            L = b[:, 1] - b[:, 0]
            if not np.any(xyz):                            # step 0: fix ave/atom not yet filled
                k += 9 + n
                continue
            at = Atoms(symbols=list(el), positions=(xyz - b[:, 0]) % L, cell=L, pbc=True)
            frames.append((step, at))
            k += 9 + n
        else:
            k += 1
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--elements", nargs="+", required=True)
    ap.add_argument("--states", nargs="+", required=True, help="label:file.data ...")
    ap.add_argument("--crack", default=None, help="folder with crack_side.txt + crack_meta.json")
    ap.add_argument("--traj", default=None, help="time-averaged heal dump for IBD(t)")
    ap.add_argument("--dt_ps", type=float, default=0.001, help="timestep (ps) to convert dump steps")
    ap.add_argument("--out", default="r1")
    a = ap.parse_args()
    Z = {k + 1: atomic_numbers[e] for k, e in enumerate(a.elements)}
    meta, side = None, None
    if a.crack:
        meta = json.load(open(os.path.join(a.crack, "crack_meta.json")))
        side = np.loadtxt(os.path.join(a.crack, "crack_side.txt"), dtype=int)
    rows, crows, summ = [], [], {}
    for spec in a.states:
        lab, fn = spec.split(":", 1)
        at = read(fn, format="lammps-data", Z_of_type=Z, atom_style="atomic")
        m, info = metrics(at)
        summ[lab] = dict(metrics=m, **info)
        for k, v in m.items():
            rows.append(dict(state=lab, interface=k, **{q: round(w, 4) if isinstance(w, float) else w for q, w in v.items()}))
        if meta is not None and len(side) == len(at):
            pl = meta["plane_frac_cracked"] if lab.startswith("cracked") else meta["plane_frac_pristine"]
            cp = crack_plane(at, side, pl, meta["axis"])
            summ[lab]["crack_plane_nm2"] = cp
            crows.append(dict(state=lab, **{k: round(v, 4) for k, v in cp.items()}))
    import csv
    with open(a.out + "_interfaces.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if crows:
        with open(a.out + "_crackplane.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(crows[0]))
            w.writeheader()
            w.writerows(crows)
    # recovery at the cracked interface I1
    if "pristine" in summ:
        p = summ["pristine"]["metrics"]["I1"]
        for lab in summ:
            q = summ[lab]["metrics"]["I1"]
            summ[lab]["I1_recovery_vs_pristine"] = {k: (q[k] / p[k] if p[k] else None) for k in ("IBD_nm2", "P_ZnOP_pct", "ICC_pct")}
    if a.traj and os.path.exists(a.traj):
        with open(a.out + "_ibd_t.csv", "w") as fh:
            fh.write("time (ps),IBD I1 (nm-2),IBD I2 (nm-2),P_ZnOP I1 (%),P_ZnOP I2 (%),ICC I1 (%),ICC I2 (%)\n")
            for step, at in read_avg_dump(a.traj, a.elements):
                m, _ = metrics(at)
                fh.write(f"{step * a.dt_ps:.2f},{m['I1']['IBD_nm2']:.4f},{m['I2']['IBD_nm2']:.4f},{m['I1']['P_ZnOP_pct']:.3f},"
                         f"{m['I2']['P_ZnOP_pct']:.3f},{m['I1']['ICC_pct']:.2f},{m['I2']['ICC_pct']:.2f}\n")
    json.dump(summ, open(a.out + "_summary.json", "w"), indent=1)
    print(f"{'state':12s} {'if':5s} {'ZnOP':>4s} {'IBD/nm2':>7s} {'P_ZnOP%':>7s} {'ICC%':>6s}")
    for r in rows:
        print(f"{r['state']:12s} {r['interface']:5s} {r['N_ZnOP']:4d} {r['IBD_nm2']:7.2f} {r['P_ZnOP_pct']:7.1f} {r['ICC_pct']:6.1f}")
    for r in crows:
        print("crack plane (per nm2):", r)


if __name__ == "__main__":
    main()
