#!/usr/bin/env python3
"""zn_environment.py -- Zn-centred partial g(r) and coordination numbers (paper Fig. 6D) and the
X-ray-weighted change of the Zn environment, composite minus parent ZIF glass (Fig. 6B).

  python post/zn_environment.py <RUN> [--out analysis] [--exp residual.csv --exp_col "residual COM 1:5"]

Structures (300 K NVT sampling frames, md_stats*.lammpstrj):
  zif        zif221_B/md_stats.lammpstrj
  pristine   composite_B/md_stats.lammpstrj
  cracked    heal_B_T*_s*/cracked.data (single as-cut structure)
  healed     heal_B_T<T>_s<seed>/md_stats_healed.lammpstrj
Writes
  <out>/zn_gr.csv        g_ZnX(r) for X = O, P, N, C, Na (bin 0.02 A); healed: per seed and mean per T
  <out>/zn_cn.csv        coordination per Zn: Zn-N, Zn-O (< 2.5 A), Zn-P (< 3.6 A), Zn-C (< 3.4), Zn-Na (< 3.8)
  <out>/zn_dDr.csv       Delta D_Zn(r) = sum_X Z_Zn Z_X 4 pi r rho_X(r) [composite - ZIF], broadened
                         with sigma = 0.6 pi / Qmax (Qmax 20 A-1); per neighbour species and total.
With --exp (CSV: first column r, a column whose header starts with --exp_col) the MD curves are
scaled by least squares to the experimental residual over 1.7-3.8 A and the Pearson r is printed.
"""
import argparse
import csv
import glob
import os

import numpy as np
from ase.io import read
from ase.neighborlist import neighbor_list

from common import heal_runs, read_data

RMAX, DR = 8.0, 0.02
EDGES = np.arange(0, RMAX + DR / 2, DR); RC = 0.5 * (EDGES[1:] + EDGES[:-1])
PARTNERS = ("O", "P", "N", "C", "Na", "H", "Zn")
ZNUM = {"C": 6, "H": 1, "N": 7, "Na": 11, "O": 8, "P": 15, "Zn": 30}


def frames(fn):
    if fn.endswith(".data"):
        return [read_data(fn)]
    fr = read(fn, index=":", format="lammps-dump-text")
    for a in fr:
        a.pbc = True; a.wrap()
    return fr


def zn_partials(fns):
    """local number density of X around Zn, n_X(r) / (4 pi r^2 dr) [A^-3], and g_ZnX(r)"""
    acc = {b: np.zeros(len(RC)) for b in PARTNERS}; nzn = 0; rho = {b: [] for b in PARTNERS}
    for fn in fns:
        for a in frames(fn):
            s = np.array(a.get_chemical_symbols()); i, j, d = neighbor_list("ijd", a, RMAX)
            m = s[i] == "Zn"; nzn += (s == "Zn").sum(); V = a.get_volume()
            for b in PARTNERS:
                h, _ = np.histogram(d[m & (s[j] == b)], EDGES); acc[b] += h; rho[b].append((s == b).sum() / V)
    shell = 4 * np.pi * RC ** 2 * DR
    loc = {b: acc[b] / nzn / shell for b in PARTNERS}
    g = {b: loc[b] / np.mean(rho[b]) if np.mean(rho[b]) > 0 else 0 * RC for b in PARTNERS}
    return loc, g


def cn(loc, b, rc):
    k = RC < rc
    return np.sum(loc[b][k] * 4 * np.pi * RC[k] ** 2 * DR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--exp", default=None, help="CSV with the experimental residual Delta D(r)")
    ap.add_argument("--exp_col", default="residual COM 1:5")
    ap.add_argument("--qmax", type=float, default=20.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = heal_runs(a.run)
    res = {"zif": zn_partials([os.path.join(a.run, "zif221_B", "md_stats.lammpstrj")]),
           "pristine": zn_partials([os.path.join(a.run, "composite_B", "md_stats.lammpstrj")])}
    if runs:
        res["cracked"] = zn_partials([os.path.join(runs[sorted(runs)[0]], "cracked.data")])
    for (T, s), d in sorted(runs.items()):
        f = glob.glob(os.path.join(d, "md_stats_healed.lammpstrj"))
        if f:
            res[f"healed_{T}_s{s}"] = zn_partials(f)
    for T in sorted({T for T, _ in runs}):
        ks = [k for k in res if k.startswith(f"healed_{T}_s")]
        if ks:
            res[f"healed_{T}_mean"] = tuple({b: np.mean([res[k][t][b] for k in ks], 0) for b in PARTNERS} for t in (0, 1))
    with open(os.path.join(a.out, "zn_gr.csv"), "w", newline="") as f:
        w = csv.writer(f); keys = list(res)
        w.writerow(["r (A)"] + [f"g Zn-{b} {k}" for k in keys for b in ("O", "P", "N", "C", "Na")])
        for i in range(len(RC)):
            w.writerow([round(RC[i], 3)] + [round(res[k][1][b][i], 4) for k in keys for b in ("O", "P", "N", "C", "Na")])
    with open(os.path.join(a.out, "zn_cn.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["structure", "Zn-N (<2.5 A)", "Zn-O (<2.5 A)", "Zn-P (<3.6 A)", "Zn-C (<3.4 A)", "Zn-Na (<3.8 A)"])
        for k, (loc, _) in res.items():
            r = [round(cn(loc, "N", 2.5), 3), round(cn(loc, "O", 2.5), 3), round(cn(loc, "P", 3.6), 3),
                 round(cn(loc, "C", 3.4), 3), round(cn(loc, "Na", 3.8), 3)]
            w.writerow([k] + r); print(f"{k:18s} CN Zn-N {r[0]:.2f}  Zn-O {r[1]:.2f}  Zn-P {r[2]:.2f}")
    # ---- Fig. 6B: X-ray-weighted change of the Zn environment, composite - ZIF
    sig = 0.6 * np.pi / a.qmax
    rr = np.arange(1.0, 6.0, 0.01)
    kern = np.exp(-0.5 * ((rr[:, None] - rr[None, :]) / sig) ** 2); kern /= kern.sum(1)[:, None]
    env = lambda k, b: ZNUM["Zn"] * ZNUM[b] * 4 * np.pi * RC * res[k][0][b]
    dE = {b: kern @ np.interp(rr, RC, env("pristine", b) - env("zif", b)) for b in ZNUM}
    dT = sum(dE.values()); scl = 1.0; hdr_x = ""
    if a.exp:
        rows = [l for l in open(a.exp) if not l.startswith("#")]
        h = rows[0].strip().split(","); X = np.array([[float(v) for v in l.split(",")] for l in rows[1:]])
        ic = [i for i, c in enumerate(h) if c.startswith(a.exp_col)][0]
        e = np.interp(rr, X[:, 0], X[:, ic]); m = (rr > 1.7) & (rr < 3.8)
        scl = np.dot(e[m], dT[m]) / np.dot(dT[m], dT[m])
        print(f"Fig 6B scale {scl:.6f}, Pearson r {np.corrcoef(e[m], dT[m])[0, 1]:.2f} vs '{a.exp_col}'")
        hdr_x = " (scaled to experiment)"
    with open(os.path.join(a.out, "zn_dDr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["r (A)"] + [f"dD Zn-{b}{hdr_x}" for b in ("O", "P", "N", "C", "Na")] + [f"dD total{hdr_x}"])
        for i in np.where(rr <= 4.0)[0]:
            w.writerow([round(rr[i], 3)] + [round(scl * dE[b][i], 5) for b in ("O", "P", "N", "C", "Na")] + [round(scl * dT[i], 5)])
    print("written to", a.out)


if __name__ == "__main__":
    main()
