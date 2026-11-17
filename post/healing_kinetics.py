#!/usr/bin/env python3
"""healing_kinetics.py -- interface re-bonding vs time during healing, from the 1-ps
time-averaged dumps (heal_avg_healed.lammpstrj) of every heal_B_T<T>_s<seed> run.

  python post/healing_kinetics.py <RUN> [--out analysis]

Writes
  <out>/kinetics_T<T>_s<seed>.csv  time, T, Lz, Zn-O-P at I1 and I2, P_ZnOP and ICC at I1,
                                   Zn-O-P and Na-O across the crack plane (every 1 ps)
  <out>/kinetics_mean.csv          mean and sd over seeds per T on the common time grid
The crack plane is followed at its distance from the ZIF centre in the cracked cell.
"""
import argparse
import csv
import json
import os

import numpy as np

from common import DT_PS, EL, heal_runs
from heal_analysis_r1 import crack_plane, metrics, read_avg_dump

COLS = ["time (ps)", "T (K)", "Lz (A)", "Zn-O-P I1", "Zn-O-P I2", "P_ZnOP I1 (%)", "ICC I1 (%)",
        "Zn-O-P across crack plane", "Na-O across crack plane"]


def one_run(d):
    meta = json.load(open(os.path.join(d, "crack_meta.json")))
    side = np.loadtxt(os.path.join(d, "crack_side.txt"), dtype=int)
    cur = np.loadtxt(os.path.join(d, "heal_curve_healed.dat"))      # step T P PE dens Lx Ly Lz
    out = []
    for step, at in read_avg_dump(os.path.join(d, "heal_avg_healed.lammpstrj"), EL):
        m, info = metrics(at); A = info["area_nm2"]
        cp = crack_plane(at, side, meta["plane_frac_cracked"], meta["axis"])
        row = cur[np.argmin(abs(cur[:, 0] - step))]
        out.append([round(step * DT_PS, 2), round(row[1], 1), round(row[7], 2), m["I1"]["N_ZnOP"], m["I2"]["N_ZnOP"],
                    round(m["I1"]["P_ZnOP_pct"], 2), round(m["I1"]["ICC_pct"], 1),
                    round(cp["Zn-O-P"] * A), round(cp["Na-O"] * A)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for (T, s), d in sorted(heal_runs(a.run).items()):
        res[(T, s)] = one_run(d)
        with open(os.path.join(a.out, f"kinetics_T{T}_s{s}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(COLS); w.writerows(res[(T, s)])
        print(T, s, len(res[(T, s)]), "frames; final across-plane Zn-O-P", res[(T, s)][-1][7])
    blocks = {}
    for T in sorted({T for T, _ in res}):
        R = [np.array(res[k], float) for k in sorted(res) if k[0] == T]
        n = min(len(r) for r in R); Y = np.stack([r[:n] for r in R])
        blocks[T] = (Y[0, :, 0], Y.mean(0), Y.std(0, ddof=1) if len(R) > 1 else 0 * Y[0])
    with open(os.path.join(a.out, "kinetics_mean.csv"), "w", newline="") as f:
        w = csv.writer(f)
        hdr = []
        for T in blocks:
            hdr += [f"time {T} K (ps)"] + [f"{c} {T} K {x}" for c in COLS[1:] for x in ("mean", "sd")]
        w.writerow(hdr)
        n = max(len(b[0]) for b in blocks.values())
        for i in range(n):
            row = []
            for T, (t, mu, sd) in blocks.items():
                row += ([t[i]] + [round(v, 3) for j in range(1, len(COLS)) for v in (mu[i, j], sd[i, j])]) if i < len(t) \
                    else [""] * (1 + 2 * (len(COLS) - 1))
            w.writerow(row)
    print("written to", a.out)


if __name__ == "__main__":
    main()
