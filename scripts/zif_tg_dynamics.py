#!/usr/bin/env python3
"""
zif_tg_dynamics.py -- glass transition of the 2x2x2 ZIF-62 (2368 atoms) quenched at 1 GPa:
thermodynamic (V, PE change points from tg_bayes.py CSV) and dynamic (Zn-N bond exchange and
Zn mobility per 10 ps from quench.lammpstrj) criteria.

  python zif_tg_dynamics.py <run_dir> <tg_cooling_csv> <out_prefix>
writes <out_prefix>.png/.pdf and <out_prefix>_dynamics.csv
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import xrd_compare as X  # noqa: E402

RUN, TGCSV, OUT = sys.argv[1:4]
fr = X.read_dump(f"{RUN}/quench.lammpstrj")
cc = np.loadtxt(f"{RUN}/cooling_curve.dat")
NSTEP = 10000                     # dump interval (10 ps at dt = 1 fs)


def bonds(el, x, L):
    zn, n = np.where(el == "Zn")[0], np.where(el == "N")[0]
    t = cKDTree(x[n] % L, boxsize=L)
    return {(i, n[j]) for i, p in zip(zn, x[zn] % L) for j in t.query_ball_point(p, 2.5)}


B = [bonds(*f) for f in fr]
rows = []
for k in range(1, len(fr)):
    s = k * NSTEP
    m = (cc[:, 0] > s - NSTEP) & (cc[:, 0] <= s)
    el, x0, L0 = fr[k - 1]
    _, x1, L1 = fr[k]
    d = x1 / L1 - x0 / L0
    d -= np.round(d)
    d *= L1
    zn = el == "Zn"
    rows.append((s / 1000, cc[m, 1].mean(), cc[m, 2].mean(), len(B[k - 1] - B[k]) / len(B[k - 1]),
                 (d[zn] ** 2).sum(1).mean(), len(B[k]) / zn.sum()))
dyn = np.array(rows)
np.savetxt(OUT + "_dynamics.csv", dyn, delimiter=",", fmt="%.5g", comments="",
           header="time (ps),T (K),P (bar),fraction of Zn-N bonds broken per 10 ps,Zn MSD per 10 ps (A^2),Zn-N coordination")
q = dyn[dyn[:, 2] > 5000]          # quench at 1 GPa only (exclude the 300 K release)

hdr = open(TGCSV).read().splitlines()
tgline = hdr[1]
tgPE = float(tgline.split("Tg(PE) = ")[1].split(" K")[0])
lo, hi = [float(v) for v in tgline.split("Tg(PE) = ")[1].split("[")[1].split("]")[0].split("-")]
d = np.loadtxt(TGCSV, delimiter=",", skiprows=3)
T, V, PE, PEfit = d[:, 0], d[:, 1], d[:, 3], d[:, 4]

fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
a = ax[0]
p = np.polyfit(T, V, 1)
a.plot(T, V, "o", ms=3.5, color="#8a8984", mec="none", label="MD (2 ps avg)")
a.plot(T, np.polyval(p, T), color="#2a78d6", lw=1.6, label="single line")
a.set_xlabel("T (K)")
a.set_ylabel("V / atom (Å$^3$)")
a.set_title("A  Volume at 1 GPa: no resolvable kink", loc="left", fontsize=10)
a.legend(frameon=False, fontsize=8)
a = ax[1]
a.plot(T, PE, "o", ms=3.5, color="#8a8984", mec="none", label="MD (2 ps avg)")
g = T <= tgPE
a.plot(T[g], PEfit[g], color="#2a78d6", lw=2, label="glass")
a.plot(T[~g], PEfit[~g], color="#eb6834", lw=2, label="liquid")
a.axvspan(lo, hi, color="#2a78d6", alpha=0.13, lw=0, label="95 % CI")
a.axvline(tgPE, color="k", ls="--", lw=1.2, label=f"Tg(PE) = {tgPE:.0f} K")
a.axvline(593, color="#b07800", ls=":", lw=1.5, label="exp Tg 593 K")
a.set_xlabel("T (K)")
a.set_ylabel("PE / atom (eV)")
a.set_title("B  Potential energy", loc="left", fontsize=10)
a.legend(frameon=False, fontsize=7.5)
a = ax[2]
a.plot(q[:, 1], 100 * q[:, 3], "o-", color="#d62728", lw=1.4, ms=4, label="Zn–N bonds broken per 10 ps")
a.set_xlabel("T (K)")
a.set_ylabel("Zn–N bonds broken per 10 ps (%)", color="#d62728")
a2 = a.twinx()
a2.plot(q[:, 1], q[:, 4], "s-", color="#1f5fbf", lw=1.2, ms=3.5, label="Zn MSD per 10 ps")
a2.set_ylabel("Zn MSD per 10 ps (Å$^2$)", color="#1f5fbf")
a.axvspan(lo, hi, color="#2a78d6", alpha=0.13, lw=0)
a.axvline(tgPE, color="k", ls="--", lw=1.2)
a.invert_xaxis()
a.set_title("C  Network dynamics during the quench", loc="left", fontsize=10)
for a_ in ax:
    a_.grid(alpha=0.25, lw=0.6)
fig.suptitle("gZ62, 2×2×2 cell (2368 atoms), MACE-MH-1, quenched at 1 GPa (5 K/ps)", fontsize=10, y=1.0)
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=300, bbox_inches="tight")
fig.savefig(OUT + ".pdf", bbox_inches="tight")
for r in q:
    print(f"T {r[1]:7.1f}  broken {100 * r[3]:5.1f} %  MSD {r[4]:.2f}")
