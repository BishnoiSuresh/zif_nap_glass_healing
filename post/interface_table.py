#!/usr/bin/env python3
"""interface_table.py -- interface bonding of every final structure of a RUN directory
(pristine, cracked, contact after 5 ps at 300 K, healed), one row per structure.

  python post/interface_table.py <RUN> [--out analysis]

Writes
  <out>/interface_metrics_all_states.csv   per structure: Zn-O-P bonds, IBD (per nm2), Zn atoms,
                                           connected Zn, ICC (%), P_ZnOP (%) for the crack plane,
                                           I1 (cracked interface), I2 (intact) and total; Na-O and
                                           P-O across the crack plane; density, Lz, min density at I1
  <out>/interface_metrics_summary_by_T.csv mean and sd over seeds of the healed structures per T
  <out>/density_profiles.csv               ZIF-centred density profile along z (1 A bins) per structure

Definitions (scripts/heal_analysis_r1.py): Zn-O <= 2.5 A, Zn-N <= 2.5 A, P-O <= 2.05 A;
Zn-O-P = Zn-O bond whose O is bonded to P; IBD = Zn-O-P per nm2 of interface ('total' uses
both interfaces, i.e. 2 x area); P_ZnOP = Zn-O-P / (Zn-N + Zn-O); ICC = % Zn with >= 1 Zn-O-P.
Zn are split into I1 / I2 halves by the circular mean of Zn along z. 'Across the crack plane'
uses the side labels written by make_crack.py (crack_side.txt, crack_meta.json).
"""
import argparse
import csv
import json
import os

import numpy as np

from common import heal_runs, mean_sd, read_data, density_profile
from heal_analysis_r1 import crack_plane, metrics


def row_for(lab, T, seed, fn, side, meta):
    at = read_data(fn)
    m, info = metrics(at)
    A = info["area_nm2"]
    r = dict(state=lab, T_heal_K=T, seed=seed, area_nm2=round(A, 3))
    pl = meta["plane_frac_pristine"] if lab == "pristine" else meta["plane_frac_cracked"]
    cp = crack_plane(at, side, pl, meta["axis"])
    r["crack plane Zn-O-P bonds"] = round(cp["Zn-O-P"] * A)
    r["crack plane IBD (Zn-O-P per nm2)"] = round(cp["Zn-O-P"], 3)
    r["crack plane Na-O bonds"] = round(cp["Na-O"] * A)
    r["crack plane P-O bonds"] = round(cp["O-P"] * A)
    for k in ("I1", "I2", "total"):
        q = m[k]; Aeff = A * (2 if k == "total" else 1)
        r[f"{k} Zn-O-P bonds"] = q["N_ZnOP"]
        r[f"{k} IBD (Zn-O-P per nm2)"] = round(q["N_ZnOP"] / Aeff, 3)
        r[f"{k} Zn atoms"] = q["Zn"]
        r[f"{k} connected Zn"] = q["Zn_with_ZnOP"]
        r[f"{k} ICC (%)"] = round(q["ICC_pct"], 2)
        r[f"{k} connected Zn per nm2"] = round(q["Zn_with_ZnOP"] / Aeff, 3)
        r[f"{k} P_ZnOP (%)"] = round(q["P_ZnOP_pct"], 2)
    L = at.cell.lengths()
    c, rho = density_profile(at.get_chemical_symbols(), at.positions[:, 2], at.get_masses(), L, dz=1.0)
    k = (c < -6) & (c > -16)
    r["density (g/cm3)"] = round(at.get_masses().sum() * 1.66054 / at.get_volume(), 3)
    r["Lz (A)"] = round(L[2], 2)
    r["min density at I1 (g/cm3)"] = round(rho[k].min(), 2)
    return r, (c, rho)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="analysis")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = heal_runs(a.run)
    if not runs:
        raise SystemExit(f"no heal_B_T*_s* folders in {a.run}")
    ref = runs[sorted(runs)[0]]              # the crack is deterministic: any heal folder gives the reference
    meta = json.load(open(os.path.join(ref, "crack_meta.json")))
    side = np.loadtxt(os.path.join(ref, "crack_side.txt"), dtype=int)
    states = [("pristine", "", "", os.path.join(a.run, "composite_B", "composite_glass.data")),
              ("cracked", "", "", os.path.join(ref, "cracked.data")),
              ("contact", "", "", os.path.join(runs.get((800, 1), ref), "start_300K_healed.data"))]   # paper: 800 K, seed 1
    for (T, s), d in sorted(runs.items()):
        states.append(("healed", T, s, os.path.join(d, "healed.data")))
    rows, profs = [], []
    for lab, T, s, fn in states:
        if not os.path.exists(fn):
            print("skip (missing)", fn); continue
        r, p = row_for(lab, T, s, fn, side, meta)
        rows.append(r); profs.append((f"{lab} {T} {s}".strip(), p))
        print(f"{lab:22s} {T!s:>4s} {s!s:>2s}  across {r['crack plane Zn-O-P bonds']:3d}  "
              f"I1 IBD {r['I1 IBD (Zn-O-P per nm2)']:.2f}  ICC {r['I1 ICC (%)']:.1f}  P_ZnOP {r['I1 P_ZnOP (%)']:.1f}")
    keys = list(rows[0])
    with open(os.path.join(a.out, "interface_metrics_all_states.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    num = [k for k in keys if k not in ("state", "T_heal_K", "seed")]
    with open(os.path.join(a.out, "interface_metrics_summary_by_T.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["T_heal_K", "n_seeds"] + [f"{k} {x}" for k in num for x in ("mean", "sd")])
        for T in sorted({T for T, _ in runs}):
            H = [r for r in rows if r["state"] == "healed" and r["T_heal_K"] == T]
            w.writerow([T, len(H)] + [round(v, 3) for k in num for v in mean_sd([h[k] for h in H])])
    with open(os.path.join(a.out, "density_profiles.csv"), "w", newline="") as f:
        c = profs[0][1][0]
        w = csv.writer(f); w.writerow(["z - z_ZIF (A)"] + [f"rho {lab} (g/cm3)" for lab, _ in profs])
        for i in range(len(c)):
            w.writerow([round(c[i], 2)] + [round(np.interp(c[i], p[0], p[1]), 4) for _, p in profs])
    print("written to", a.out)


if __name__ == "__main__":
    main()
