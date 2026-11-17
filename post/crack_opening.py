#!/usr/bin/env python3
"""crack_opening.py -- crack opening (interfacial void width at I1) over the full healing
cycle of every heal_B_T<T>_s<seed> run, with the temperature (paper Fig. 6C).

  python post/crack_opening.py <RUN> [--out analysis] [--plot]

Void width = sum over 0.5 A slices, -20 A < z - z_ZIF < -3 A, of dz * max(0, 1 - rho/1.5 g cm-3)
(ZIF-centred mass-density profile; pristine composite ~0.4 A, as-cut crack ~4.9 A).

Sources per phase
  t = 0                         cracked.data (as cut)
  heating + hold                heal_avg_healed.lammpstrj   (1-ps time averages)
  quench + 300 K anneal         heal_healed.lammpstrj       (instantaneous, every 5 ps)
  final 300 K sampling          md_stats_healed.lammpstrj   (instantaneous)
Writes <out>/crack_opening_T<T>_s<seed>.csv, <out>/crack_opening_mean.csv
(+ <out>/crack_opening.png with --plot). Also prints pristine / cracked / healed references.
"""
import argparse
import csv
import os

import numpy as np

from common import DT_PS, EL, MASS, heal_runs, iter_dump, read_data, void_width
from heal_analysis_r1 import read_avg_dump


def vw_data(fn):
    at = read_data(fn)
    return void_width(at.get_chemical_symbols(), at.positions[:, 2], at.get_masses(), at.cell.lengths())[0], at.cell.lengths()[2]


def vw_dump(fn):
    for step, L, el, xyz in iter_dump(fn):
        m = np.array([MASS[e] for e in el])
        yield step, L[2], void_width(el, xyz[:, 2], m, L)[0]


def one_run(d):
    cur = np.loadtxt(os.path.join(d, "heal_curve_healed.dat"))     # step T P PE dens Lx Ly Lz
    Tat = lambda step: cur[np.argmin(abs(cur[:, 0] - step)), 1]
    rows = [(0.0, 300.0, vw_data(os.path.join(d, "cracked.data"))[0], "cracked (t = 0)")]
    for step, at in read_avg_dump(os.path.join(d, "heal_avg_healed.lammpstrj"), EL):
        vw = void_width(at.get_chemical_symbols(), at.positions[:, 2], at.get_masses(), at.cell.lengths())[0]
        rows.append((step * DT_PS, Tat(step), vw, "heat/hold, 1-ps averaged"))
    tend = rows[-1][0]
    stats = os.path.join(d, "md_stats_healed.lammpstrj")
    first_stats = next(iter_dump(stats))[0] if os.path.exists(stats) else np.inf
    inst = os.path.join(d, "heal_healed.lammpstrj")
    if os.path.exists(inst):
        for step, Lz, vw in vw_dump(inst):
            if step * DT_PS > tend + 0.5 and step < first_stats:
                rows.append((step * DT_PS, Tat(step), vw, "quench/anneal, instantaneous"))
    if os.path.exists(stats):
        for step, Lz, vw in vw_dump(stats):
            rows.append((step * DT_PS, Tat(step), vw, "final 300 K sampling, instantaneous"))
    return rows, cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = heal_runs(a.run)
    pr = vw_data(os.path.join(a.run, "composite_B", "composite_glass.data"))
    print(f"reference void width: pristine {pr[0]:.2f} A (Lz {pr[1]:.2f} A)")
    full = {}
    for (T, s), d in sorted(runs.items()):
        rows, cur = one_run(d); full[(T, s)] = (rows, cur)
        with open(os.path.join(a.out, f"crack_opening_T{T}_s{s}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["time (ps)", "T (K)", "interfacial void width (A)", "phase"])
            for r in rows:
                w.writerow([round(r[0], 2), round(r[1], 1), round(r[2], 3), r[3]])
        print(f"T {T} s{s}: cracked {rows[0][2]:.2f} A -> final {rows[-1][2]:.2f} A ({len(rows)} points)")
    Ts = sorted({T for T, _ in full}); blocks = {}
    for T in Ts:
        R = [full[k][0] for k in sorted(full) if k[0] == T]; n = min(len(r) for r in R)
        t = np.array([R[0][i][0] for i in range(n)]); y = np.array([[r[i][2] for i in range(n)] for r in R])
        cur = full[(T, min(s for TT, s in full if TT == T))][1]
        Tc = np.mean([full[k][1][:len(cur), 1] for k in full if k[0] == T], 0)
        blocks[T] = (t, y.mean(0), y.std(0, ddof=1) if len(R) > 1 else 0 * t, cur[:, 0] * DT_PS, Tc)
    with open(os.path.join(a.out, "crack_opening_mean.csv"), "w", newline="") as f:
        w = csv.writer(f); h = []
        for T in Ts:
            h += [f"time {T} K (ps)", f"void width mean {T} K (A)", f"void width sd {T} K (A)",
                  f"time T-curve {T} K (ps)", f"T {T} K run (K)"]
        w.writerow(h); n = max(max(len(b[0]), len(b[3])) for b in blocks.values())
        for i in range(n):
            row = []
            for T in Ts:
                t, mu, sd, tc, Tc = blocks[T]
                row += [round(t[i], 2), round(mu[i], 3), round(sd[i], 3)] if i < len(t) else ["", "", ""]
                row += [round(tc[i], 1), round(Tc[i], 1)] if i < len(tc) else ["", ""]
            w.writerow(row)
    if a.plot and Ts:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, len(Ts), figsize=(5 * len(Ts), 4.2), squeeze=False)
        for axx, T in zip(ax[0], Ts):
            t, mu, sd, tc, Tc = blocks[T]
            axx.plot(t, mu, c="#1BAF7A", lw=2); axx.fill_between(t, mu - sd, mu + sd, color="#1BAF7A", alpha=.2, lw=0)
            axx.axhline(pr[0], ls="--", c="k", lw=.7)
            axx.set(xlabel="time (ps)", ylabel="interfacial void width (Å)", title=f"T_heal = {T} K", ylim=(0, 5.3))
            b = axx.twinx(); b.plot(tc, Tc, c="#C8860B", lw=1.5); b.set_ylabel("T (K)", color="#C8860B")
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "crack_opening.png"), dpi=150)
    print("written to", a.out)


if __name__ == "__main__":
    main()
