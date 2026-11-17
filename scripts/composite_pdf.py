#!/usr/bin/env python3
"""
composite_pdf.py -- interfacial signal in the X-ray PDF of NaP/gZ62 hybrid glasses,
experiment (DESY, reduced with desy_reduce.py) vs MD (xrd_compare.py functions).

For each mixture, D_mix(r) is fitted as a*D_NaP + b*D_ZIF (1 < r < 8 A, parent glasses
measured/simulated identically).  A physical mixture (PM) should leave ~no residual;
the residual of the composite (COM) = correlations that exist only in the hybrid
(Zn-O(P), Zn...P, Na...imidazolate, changed ZIF/NaP networks).
For MD the same fit is done, plus the exact 'no-interface' reference
i_ref(Q) = sum_p c_p <f>_p^2 i_p(Q) / <f>^2 and the X-ray-weighted partial
contributions of the hetero pairs (Zn-P, Zn-O, Na-N, Na-C) that only exist in the composite.

  python composite_pdf.py desy_a1.0.npz sim_x.npz out_prefix sim_zn.npz
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import periodictable.cromermann as cm

EXP, SIM, OUT = sys.argv[1:4]
z = np.load(EXP)
s = np.load(SIM, allow_pickle=True)
r = z["r"]
FIT = (r > 1.0) & (r < 8.0)


def fit2(y, a, b):
    A = np.vstack([a[FIT], b[FIT]]).T
    c, *_ = np.linalg.lstsq(A, y[FIT], rcond=None)
    ref = c[0] * a + c[1] * b
    res = y - ref
    R = np.sqrt((res[FIT] ** 2).sum() / (y[FIT] ** 2).sum())
    return c, ref, res, R


def favq(comp, q):
    n = sum(comp.values())
    fav = sum(v / n * cm.fxrayatq(k, q) for k, v in comp.items())
    return n, fav


# ---------------- experiment ----------------
Dn, Dz = z["27NaP__D"], z["gZ62__D"]
exp = {}
for lab in ["PM_1-1", "COM_1-1", "EVP_1-1", "PM_1-5", "COM_1-5", "COM_5-1"]:
    exp[lab] = fit2(z[lab + "__D"], Dn, Dz)
    c, _, _, R = exp[lab]
    print(f"exp {lab:8s}  a(NaP) {c[0]:.2f}  b(ZIF) {c[1]:.2f}  R_res {R:.3f}")

# ---------------- simulation ----------------
q, rs = s["q"], s["r"]
comp = {k: eval(str(s[k + "_comp"])) for k in ("comp", "nap", "zif")}
comp = {k: {str(e): n for e, n in v.items()} for k, v in comp.items()}
Nt, ft = favq(comp["comp"], q)
iref = 0
for p in ("nap", "zif"):
    Np, fp = favq(comp[p], q)
    # parents scaled to the atoms they contribute to the composite (NaP 2160, ZIF 592)
    frac = {"nap": sum(v for k, v in comp["comp"].items() if k in ("Na", "P", "O")),
            "zif": sum(v for k, v in comp["comp"].items() if k in ("Zn", "C", "H", "N"))}[p] / Nt
    iref = iref + frac * fp ** 2 * (s[p + "_S"] - 1) / ft ** 2
m = q <= 20.0
Dref = 2 / np.pi * np.trapezoid(q[m] * iref[m] * np.sinc(q[m] / 20.0) * np.sin(np.outer(rs, q[m])), q[m], axis=1)
Dc = s["comp_D"]
cs, Dfit, Rsfit, Rsim = fit2(Dc, s["nap_D"], s["zif_D"])
print(f"sim composite fit a {cs[0]:.2f} b {cs[1]:.2f} R_res {Rsim:.3f};  exact-ref R_res "
      f"{np.sqrt(((Dc - Dref)[FIT] ** 2).sum() / (Dc[FIT] ** 2).sum()):.3f}")

# ---- MD: change of the X-ray-weighted environment of Zn (composite - parent gZ62), per Zn ----
zn = np.load(sys.argv[4])
ZZ = {"C": 6, "H": 1, "N": 7, "Na": 11, "O": 8, "P": 15, "Zn": 30}
sig = 0.6 * np.pi / 20.0                        # ~ Lorch resolution for Qmax = 20 A^-1


def zn_env(k):
    rg = zn[k + "_rg"]
    cnt = {str(a): b for a, b in eval(str(zn[k + "_comp"])).items()}
    V = sum(cnt.values()) / float(zn[k + "_rho"])
    out = {}
    for b in ZZ:
        key = f"{k}_g_Zn{b}"
        if key in zn.files:
            out[b] = ZZ["Zn"] * ZZ[b] * 4 * np.pi * rg * cnt[b] / V * zn[key]   # first-shell terms only: compare for r < 4 A
    return rg, out


def broaden(rg, y):
    kk = np.exp(-0.5 * ((rg[:, None] - rg[None, :]) / sig) ** 2)
    return (kk @ y) / kk.sum(1)


rgc, Ec = zn_env("comp")
rgz, Ez = zn_env("zif")
rmax = min(rgc[-1], rgz[-1], 6.0)
rr = np.arange(1.0, rmax, 0.01)
dEnv = {b: broaden(rr, np.interp(rr, rgc, Ec.get(b, 0 * rgc)) - np.interp(rr, rgz, Ez.get(b, 0 * rgz)))
        for b in ZZ}
dTot = sum(dEnv.values())
cn = lambda k, b, rc: (lambda rg, cnt, V: 4 * np.pi * cnt[b] / V * np.sum(zn[f"{k}_g_Zn{b}"][rg < rc] * rg[rg < rc] ** 2) * (rg[1] - rg[0]))(
    zn[k + "_rg"], {str(a): c for a, c in eval(str(zn[k + "_comp"])).items()},
    sum(eval(str(zn[k + "_comp"])).values()) / float(zn[k + "_rho"]))
CN = {k: {"Zn-N": cn(k, "N", 2.5), "Zn-C(<3.4)": cn(k, "C", 3.4),
          "Zn-O": cn(k, "O", 2.5) if f"{k}_g_ZnO" in zn.files else 0.0,
          "Zn-P(<3.6)": cn(k, "P", 3.6) if f"{k}_g_ZnP" in zn.files else 0.0,
          "Zn-Na(<3.8)": cn(k, "Na", 3.8) if f"{k}_g_ZnNa" in zn.files else 0.0} for k in ("zif", "comp")}
for k, v in CN.items():
    print(k, {a: round(b, 2) for a, b in v.items()})
# exp COM 1:1 residual, scale MD shape to it over 1.7-3.8 A
resC = exp["COM_1-1"][2]
ei = np.interp(rr, r, resC)
w = (rr > 1.7) & (rr < 3.8)
scl = np.dot(ei[w], dTot[w]) / np.dot(dTot[w], dTot[w])
cc = np.corrcoef(ei[w], dTot[w])[0, 1]
print(f"MD Zn-environment change vs exp COM 1:1 residual (1.7-3.8 A): Pearson r = {cc:.2f}")


def pk(x, y, lo, hi, sign=1):
    m = (x > lo) & (x < hi)
    i = np.argmax(sign * y[m])
    return x[m][i]


peaks = {"Zn–O": (pk(r, resC, 1.85, 2.2), pk(rr, dEnv["O"], 1.85, 2.2)),
         "loss Zn···C": (pk(r, resC, 2.6, 3.05, -1), pk(rr, dTot, 2.6, 3.05, -1)),
         "Zn···P": (pk(r, resC, 3.05, 3.6), pk(rr, dTot, 3.05, 3.6))}
for k, (a_, b_) in peaks.items():
    print(f"  {k}: exp {a_:.2f} A, MD {b_:.2f} A")

# ---------------- figure ----------------
C = {"NaP": "#d62728", "ZIF": "#1f5fbf", "PM": "#2ca02c", "COM": "#ff8c00", "fit": "#555555"}
fig, ax = plt.subplots(2, 2, figsize=(12, 8.6))
a = ax[0, 0]
a.plot(r, Dn, c=C["NaP"], lw=1.3, label="27NaP")
a.plot(r, Dz, c=C["ZIF"], lw=1.3, label="gZ62")
a.plot(r, z["PM_1-1__D"], c=C["PM"], lw=1.5, label="PM 1:1")
a.plot(r, z["COM_1-1__D"], c=C["COM"], lw=1.5, label="COM 1:1")
a.set_title("A  Experimental X-ray D(r), gZ62:27NaP = 1:1 (wt)", loc="left", fontsize=11)
a.set_ylabel("D(r) (Å$^{-2}$)")
a.legend(frameon=False, fontsize=9, ncol=2)

a = ax[0, 1]
for lab, col in [("PM_1-1", C["PM"]), ("COM_1-1", C["COM"])]:
    c, ref, res, R = exp[lab]
    a.plot(r, z[lab + "__D"], c=col, lw=1.4, label=lab.replace("_", " ").replace("1-1", "1:1"))
    a.plot(r, ref, c=col, lw=0.9, ls="--")
    a.plot(r, res - 3.2, c=col, lw=1.6)
a.text(7.9, -2.55, f"residuals:  PM R = {exp['PM_1-1'][3]:.2f},  COM R = {exp['COM_1-1'][3]:.2f}",
       ha="right", fontsize=8.5)
a.axhline(-3.2, c="#bbbbbb", lw=0.7)
a.set_ylim(-4.3, 3.6)
a.set_title("B  Mixture, two-phase fit (dashed) and residual", loc="left", fontsize=11)
a.legend(frameon=False, fontsize=9)

a = ax[1, 0]
cols = {"O": "#d62728", "P": "#8c1aff", "N": "#1f5fbf", "C": "#555555", "Na": "#2ca02c", "Zn": "#e377c2"}
for b, col in cols.items():
    a.plot(rr, dEnv[b] / 1e3, c=col, lw=1.2, label=f"Zn–{b}")
a.plot(rr, dTot / 1e3, c="k", lw=2.0, label="total")
a.axhline(0, c="#bbbbbb", lw=0.7)
a.set_xlim(1.0, 4.0)
a.set_ylabel("Δ (Z-weighted, arb.)")
a.set_title("C  MD: Zn environment, composite − gZ62 (per Zn)", loc="left", fontsize=11)
a.text(0.98, 0.04, "CN per Zn (gZ62 → composite):\n"
       f"Zn–N {CN['zif']['Zn-N']:.2f} → {CN['comp']['Zn-N']:.2f};  Zn–O 0 → {CN['comp']['Zn-O']:.2f}\n"
       f"Zn···C {CN['zif']['Zn-C(<3.4)']:.1f} → {CN['comp']['Zn-C(<3.4)']:.1f};  Zn···P 0 → {CN['comp']['Zn-P(<3.6)']:.2f}",
       transform=a.transAxes, ha="right", va="bottom", fontsize=8)
a.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

a = ax[1, 1]
a.plot(r, resC, c=C["COM"], lw=1.8, label="exp: COM 1:1 residual")
a.plot(r, exp["PM_1-1"][2], c=C["PM"], lw=1.2, label="exp: PM 1:1 residual")
a.plot(r, exp["COM_1-5"][2], c="#b35900", lw=1.0, ls="-.", label="exp: COM 1:5 residual")
a.plot(r, exp["COM_5-1"][2], c="#ffbf66", lw=1.0, ls="-.", label="exp: COM 5:1 residual")
a.plot(rr, scl * dTot, c="k", lw=1.6, ls="--", label=f"MD Zn-environment change (scaled, r = {cc:.2f})")
for t, (xe, xm) in peaks.items():
    a.axvline(xe, c="#bbbbbb", lw=0.6, ls=":")
    a.text(xe, -1.0, t, fontsize=8, ha="center")
a.axhline(0, c="#bbbbbb", lw=0.7)
a.set_xlim(1.0, 4.0)
a.set_ylim(-1.1, 1.65)
a.set_ylabel("ΔD(r) (Å$^{-2}$)")
a.set_title("D  Interfacial signature: experiment vs MD", loc="left", fontsize=11)
a.legend(frameon=False, fontsize=7.5, loc="upper left")
for a in ax[0]:
    a.set_xlim(0.8, 8)
for a in ax.flat:
    a.set_xlabel("r (Å)")
    a.grid(alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig(OUT + ".png", dpi=200)
fig.savefig(OUT + ".pdf")
with open(OUT + "_numbers.txt", "w") as f:
    for lab, (c, _, _, R) in exp.items():
        f.write(f"exp {lab}: a_NaP {c[0]:.3f} b_ZIF {c[1]:.3f} ZIF share {c[1] / c.sum():.2f} R_res {R:.3f}\n")
    f.write(f"sim composite (whole cell) vs no-interface ref: fit R_res {Rsim:.3f} (dominated by NaP realisation noise)\n")
    for k, v in CN.items():
        f.write(f"MD CN {k}: " + ", ".join(f"{a_} {b_:.2f}" for a_, b_ in v.items()) + "\n")
    f.write(f"MD Zn-environment change vs exp COM 1:1 residual, 1.7-3.8 A: Pearson r {cc:.2f}, scale {scl:.3g}\n")
    for t, (xe, xm) in peaks.items():
        f.write(f"{t}: exp {xe:.2f} A, MD {xm:.2f} A\n")

# ---------------- optional: Origin-ready CSV of every plotted curve ----------------
if len(sys.argv) > 5:
    import os
    CD = sys.argv[5]
    os.makedirs(CD, exist_ok=True)

    def dump(fn, cols, head=""):
        n = max(len(v) for _, v in cols)
        with open(os.path.join(CD, fn), "w") as fh:
            if head:
                fh.write("# " + head + "\n")
            fh.write(",".join(c for c, _ in cols) + "\n")
            for k in range(n):
                fh.write(",".join(f"{v[k]:.6g}" if k < len(v) else "" for _, v in cols) + "\n")

    m8 = (r >= 0.8) & (r <= 8.0)
    dump("FigS_MD8A_exp_Dr_1to1.csv",
         [("r (A)", r[m8]), ("D(r) 27NaP (1/A^2)", Dn[m8]), ("D(r) gZ62", Dz[m8]),
          ("D(r) PM 1:1", z["PM_1-1__D"][m8]), ("D(r) COM 1:1", z["COM_1-1__D"][m8])],
         "DESY P02.1 X-ray D(r)=4 pi rho0 r[g(r)-1], Q 0.8-20 1/A, Lorch; gZ62:27NaP weight ratios")
    colsB = [("r (A)", r[m8])]
    for lab in ["PM_1-1", "COM_1-1", "EVP_1-1", "PM_1-5", "COM_1-5", "COM_5-1"]:
        c_, ref, res, R = exp[lab]
        L_ = lab.replace("_", " ").replace("-", ":")
        colsB += [(f"D(r) {L_}", z[lab + "__D"][m8]), (f"two-phase fit {L_} (a_NaP={c_[0]:.3f} b_ZIF={c_[1]:.3f})", ref[m8]),
                  (f"residual {L_} (R={R:.3f})", res[m8])]
    dump("FigS_MD8B_exp_twophase_fit_and_residuals.csv", colsB,
         "fit D_mix = a*D_27NaP + b*D_gZ62 over 1-8 A; residual = D_mix - fit (plotted offset by -3.2 in the figure)")
    dump("FigS_MD8C_MD_Zn_environment_change.csv",
         [("r (A)", rr)] + [(f"Zn-{b} change (Z-weighted; arb.)", dEnv[b]) for b in ["O", "P", "N", "C", "Na", "Zn", "H"]] +
         [("total change", dTot)],
         "MD composite (1200 K) minus bulk gZ62, per Zn, Z_Zn*Z_b*4 pi r rho_b g_Znb(r), Gaussian-broadened (sigma 0.094 A); "
         + "CN per Zn gZ62->composite: " + "; ".join(f"{q_} {CN['zif'][q_]:.2f}->{CN['comp'][q_]:.2f}" for q_ in CN["comp"]))
    mD = (r >= 1.0) & (r <= 4.0)
    mm = rr <= 4.0
    dump("FigS_MD8D_residuals_vs_MD.csv",
         [("r (A)", r[mD]), ("exp residual COM 1:1", exp["COM_1-1"][2][mD]), ("exp residual PM 1:1", exp["PM_1-1"][2][mD]),
          ("exp residual COM 1:5", exp["COM_1-5"][2][mD]), ("exp residual COM 5:1", exp["COM_5-1"][2][mD]),
          ("r MD (A)", rr[mm]), (f"MD Zn-environment change x {scl:.4g} (Pearson r={cc:.2f})", (scl * dTot)[mm])],
         "peak positions exp/MD: " + "; ".join(f"{k.replace('–', '-').replace('···', '...')} {a_:.2f}/{b_:.2f} A" for k, (a_, b_) in peaks.items()))
    print("CSV written to", CD)
