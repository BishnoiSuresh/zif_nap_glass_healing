#!/usr/bin/env python3
"""elastic_table.py -- moduli of every elastic_B/elastic_C_<item>.txt (lammps/in.elastic) and the
recovery figure (pristine -> cracked -> contact -> healed per T).

  python post/elastic_table.py <RUN> [--out analysis] [--plot]

Writes
  <out>/elastic_all_structures.csv     C11..C66, directional Young's moduli Ex, Ey, Ez (= 1/S_ii),
                                  Voigt-Reuss-Hill B, G, E, smallest eigenvalue of C (GPa)
  <out>/elastic_summary_by_T.csv  healed mean / sd per T and % of pristine
A cracked cell with an open gap has C33 ~ 0 / an unstable tensor (eig_min <= 0): its Ez is
reported as 0 and B, G, E (VRH) as NaN.
"""
import argparse
import csv
import glob
import os
import re

import numpy as np

from common import mean_sd


def moduli(fn):
    rows = [list(map(float, l.split())) for l in open(fn) if l.strip() and not l.startswith("#")]
    C = np.array(rows).T; C = 0.5 * (C + C.T)
    S = np.linalg.inv(C)
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    KR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    K, G = (KV + KR) / 2, (GV + GR) / 2
    eig = np.linalg.eigvalsh(C).min()
    Ed = 1 / np.diag(S)[:3]
    r = dict(C11=C[0, 0], C22=C[1, 1], C33=C[2, 2], C13=C[0, 2], C44=C[3, 3], C55=C[4, 4], C66=C[5, 5],
             Ex=Ed[0], Ey=Ed[1], Ez=Ed[2], B_VRH=K, G_VRH=G, E_VRH=9 * K * G / (3 * K + G), eig_min=eig)
    if eig <= 0:
        r["Ez"] = 0.0 if C[2, 2] < 1 or Ed[2] < 0 else Ed[2]; r["G_VRH"] = r["B_VRH"] = r["E_VRH"] = np.nan
    return {k: round(float(v), 3) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    R = {}
    for fn in sorted(glob.glob(os.path.join(a.run, "elastic_B", "elastic_C_*.txt"))):
        R[os.path.basename(fn)[10:-4]] = moduli(fn)
    if not R:
        raise SystemExit("no elastic_B/elastic_C_*.txt found")
    keys = list(next(iter(R.values())))
    with open(os.path.join(a.out, "elastic_all_structures.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["structure"] + keys)
        for k, v in R.items():
            w.writerow([k] + [v[q] for q in keys])
    P = R.get("pristine")
    Ts = sorted({int(m.group(1)) for k in R for m in [re.match(r"healed_T(\d+)_s\d+$", k)] if m})
    with open(os.path.join(a.out, "elastic_summary_by_T.csv"), "w", newline="") as f:
        w = csv.writer(f); q = ("Ez", "Ex", "Ey", "B_VRH", "G_VRH", "E_VRH")
        w.writerow(["T_heal_K", "n"] + [f"{x} {y}" for x in q for y in ("mean", "sd", "% of pristine")])
        for T in Ts:
            H = [v for k, v in R.items() if re.match(rf"healed_T{T}_s\d+$", k)]
            row = [T, len(H)]
            for x in q:
                mu, sd = mean_sd([h[x] for h in H]); row += [round(mu, 2), round(sd, 2), round(100 * mu / P[x], 1) if P else ""]
            w.writerow(row)
            print(f"T {T}: Ez {row[2]:.1f} +- {row[3]:.1f} GPa" + (f" ({row[4]} % of pristine {P['Ez']:.1f})" if P else ""))
    if a.plot and P:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cr = R.get("cracked"); co = R.get("contact")
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
        spec = (("Ez", "E_z across the interface (GPa)", lambda r: r["Ez"]),
                ("Exy", "in-plane (E_x + E_y)/2 (GPa)", lambda r: (r["Ex"] + r["Ey"]) / 2),
                ("G", "shear modulus G (VRH, GPa)", lambda r: r["G_VRH"]))
        for axx, (_, yl, f) in zip(ax, spec):
            ref = [f(P), f(cr) if cr else np.nan, max(f(co), 0) if co else np.nan]
            axx.plot([0, 1, 2], ref, "ko-", lw=1, ms=5)
            for i, T in enumerate(Ts):
                hv = np.array([f(v) for k, v in R.items() if re.match(rf"healed_T{T}_s\d+$", k)])
                axx.errorbar(3, hv.mean(), yerr=hv.std(ddof=1) if len(hv) > 1 else 0, fmt="o", color=f"C{i}", capsize=3, label=f"{T} K")
                axx.plot([3] * len(hv), hv, ".", color=f"C{i}", ms=3, alpha=.6)
            axx.axhline(f(P), ls="--", c="k", lw=.6); axx.set_xticks(range(4))
            axx.set_xticklabels(["pristine", "cracked", "contact (5 ps)", "healed"], fontsize=8); axx.set_ylabel(yl); axx.set_ylim(bottom=0)
        ax[0].legend(fontsize=8, loc="lower right"); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "elastic_recovery.png"), dpi=150)
    print("written to", a.out)


if __name__ == "__main__":
    main()
