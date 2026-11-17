#!/usr/bin/env python3
"""
tg_bayes.py -- glass transition temperature from LAMMPS cooling curves by
Bayesian change-point analysis of a continuous two-segment (glass / liquid) line.

Model for each candidate Tg:   y(T) = a + b*T + c*max(T - Tg, 0) + e,   e ~ N(0, s^2)
  a, b, c flat priors and s Jeffreys prior, all marginalised analytically:
      p(Tg | y)  ∝  |X^T X|^(-1/2) * RSS(Tg)^(-(N-3)/2)        (O Ruanaidh & Fitzgerald 1996)
  uniform prior on Tg over [--tgmin, --tgmax].
Replicas (same --group prefix) are combined as independent data sets sharing one Tg
(each keeps its own a, b, c, s):   log p(Tg | all) = sum_i log p_i(Tg).
Reported: MAP and central 95 % credible interval, for V/atom and PE/atom.

Only the NPT quench (end of the hot hold -> ~300 K) of cooling_curve.dat is used,
and only T <= --tmax (the hot liquid is curved and creates false kinks).

  python tg_bayes.py a/cooling_curve.dat:540:40NaP_s1 b/cooling_curve.dat:540:40NaP_s2 ... \
       --tmax 1300 --tgmin 400 --tgmax 1100 --group 40NaP_s,40NaP_D3 --out tg_40nap
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GRAY, INK = "#2a78d6", "#eb6834", "#8a8984", "#52514e"


def quench_segment(a, tlow=330.0):
    T = a[:, 1]
    thot = np.median(T[:5])
    i0 = np.where(T >= thot - 30)[0][-1]
    after = np.where(T[i0:] <= tlow)[0]
    i1 = i0 + (after[0] if len(after) else len(T) - i0 - 1)
    return a[i0:i1 + 1], thot


def logpost(T, y, grid):
    N = len(T)
    out = np.empty(len(grid))
    for k, tg in enumerate(grid):
        X = np.column_stack([np.ones(N), T, np.clip(T - tg, 0, None)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ beta) ** 2)
        out[k] = -0.5 * np.linalg.slogdet(X.T @ X)[1] - 0.5 * (N - 3) * np.log(rss)
    return out


def summarize(grid, lp):
    p = np.exp(lp - lp.max())
    dg = grid[1] - grid[0]
    p /= p.sum() * dg
    cdf = np.cumsum(p) * dg
    cdf /= cdf[-1]
    q = lambda f: float(np.interp(f, cdf, grid))
    return dict(post=p, map=float(grid[np.argmax(p)]), lo=q(0.025), hi=q(0.975))


def fit_line(T, y, tg):
    X = np.column_stack([np.ones(len(T)), T, np.clip(T - tg, 0, None)])
    return np.linalg.lstsq(X, y, rcond=None)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="file:natoms:label")
    ap.add_argument("--out", default="tg_summary")
    ap.add_argument("--tlow", type=float, default=330.0)
    ap.add_argument("--tmax", type=float, default=1e9, help="use only T <= tmax")
    ap.add_argument("--tgmin", type=float, default=None)
    ap.add_argument("--tgmax", type=float, default=None)
    ap.add_argument("--group", default="", help="comma list of label prefixes combined into one joint Tg")
    ap.add_argument("--exp", default="", help="label_prefix=Tg_exp,... e.g. 40NaP=550,ZIF=593")
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--csvdir", default=None, help="write the plotted data (Origin-ready CSV) here")
    ap.add_argument("--csvprefix", default="Tg")
    a = ap.parse_args()
    exp = {k: float(v) for k, v in (x.split("=") for x in a.exp.split(",") if x)}
    groups = [g for g in a.group.split(",") if g]

    data = []
    for spec in a.runs:
        f, nat, lab = spec.split(":")
        d, thot = quench_segment(np.loadtxt(f), a.tlow)
        d = d[d[:, 1] <= a.tmax]
        data.append(dict(lab=lab, nat=int(nat), thot=thot, T=d[:, 1],
                         V=d[:, 4] / int(nat), PE=d[:, 3] / int(nat), rho=d[:, 5]))
    tlo = max(x["T"].min() for x in data)
    thi = min(x["T"].max() for x in data)
    g0 = a.tgmin if a.tgmin else tlo + 0.12 * (thi - tlo)
    g1 = a.tgmax if a.tgmax else thi - 0.12 * (thi - tlo)
    grid = np.linspace(g0, g1, 700)

    for x in data:
        for key in ("V", "PE"):
            x["lp" + key] = logpost(x["T"], x[key], grid)
            x[key + "s"] = summarize(grid, x["lp" + key])

    rows = []
    n = len(data)
    ncol = min(a.ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow), squeeze=False)
    for ax, x in zip(axes.flat, data):
        s = x["Vs"]
        b = fit_line(x["T"], x["V"], s["map"])
        f_ = lambda t: b[0] + b[1] * t + b[2] * np.clip(t - s["map"], 0, None)
        ax.plot(x["T"], x["V"], "o", ms=3.5, color=GRAY, alpha=0.7, mec="none", label="MD (2 ps avg)")
        tl = np.linspace(x["T"].min(), s["map"], 50)
        th = np.linspace(s["map"], x["T"].max(), 50)
        ax.plot(tl, f_(tl), color=BLUE, lw=2, label="glass")
        ax.plot(th, f_(th), color=ORANGE, lw=2, label="liquid")
        ax.axvspan(s["lo"], s["hi"], color=BLUE, alpha=0.13, lw=0, label="95 % CI")
        ax.axvline(s["map"], color="k", ls="--", lw=1.2, label=f"Tg(V) = {s['map']:.0f} K")
        for k, v in exp.items():
            if x["lab"].startswith(k):
                ax.axvline(v, color="#b07800", ls=":", lw=1.5, label=f"exp Tg ≈ {v:.0f} K")
        ax.set_title(f"{x['lab']}  ({x['nat']} at., THOT {x['thot']:.0f} K)", fontsize=9)
        ax.set_xlabel("T (K)")
        ax.set_ylabel("V / atom (Å$^3$)")
        ax.grid(alpha=0.25, lw=0.6)
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        ins = ax.inset_axes([0.60, 0.08, 0.37, 0.28])
        ins.plot(grid, s["post"], color=BLUE, lw=1.2)
        ins.plot(grid, x["PEs"]["post"], color=ORANGE, lw=1.2)
        ins.set_yticks([])
        ins.tick_params(labelsize=6)
        ins.set_title("p(Tg): V blue, PE orange", fontsize=6, color=INK)
        rows.append(dict(run=x["lab"], natoms=x["nat"], THOT=round(x["thot"]),
                         Tg_V=round(s["map"]), V_lo95=round(s["lo"]), V_hi95=round(s["hi"]),
                         Tg_PE=round(x["PEs"]["map"]), PE_lo95=round(x["PEs"]["lo"]), PE_hi95=round(x["PEs"]["hi"]),
                         alpha_glass_1e6=round(b[1] / f_(300.0) * 1e6, 1),
                         alpha_liquid_1e6=round((b[1] + b[2]) / f_(s["map"]) * 1e6, 1)))
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(a.out + ".png", dpi=170)

    # joint (replica-combined) posteriors
    joint = []
    if groups:
        fig2, ax2 = plt.subplots(1, 2, figsize=(10, 3.4))
        cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
        for gi, g in enumerate(groups):
            mem = [x for x in data if x["lab"].startswith(g)]
            if not mem:
                continue
            for j, key in enumerate(("V", "PE")):
                s = summarize(grid, sum(x["lp" + key] for x in mem))
                ax2[j].plot(grid, s["post"], color=cols[gi % len(cols)], lw=2,
                            label=f"{g}* (n={len(mem)}): {s['map']:.0f} K [{s['lo']:.0f}-{s['hi']:.0f}]")
                joint.append(dict(group=g, n=len(mem), prop=key, Tg=round(s["map"]), lo95=round(s["lo"]), hi95=round(s["hi"])))
        for j, key in enumerate(("V/atom", "PE/atom")):
            ax2[j].set_title(f"Joint posterior p(Tg) from {key}", fontsize=10)
            ax2[j].set_xlabel("Tg (K)")
            ax2[j].set_yticks([])
            ax2[j].grid(alpha=0.25, lw=0.6)
            ax2[j].legend(fontsize=7, frameon=False)
            for k, v in exp.items():
                ax2[j].axvline(v, color="#b07800", ls=":", lw=1.2)
        fig2.tight_layout()
        fig2.savefig(a.out + "_joint.png", dpi=170)

    if a.csvdir:
        import os
        os.makedirs(a.csvdir, exist_ok=True)
        for x, r in zip(data, rows):
            s = x["Vs"]
            b = fit_line(x["T"], x["V"], s["map"])
            fit = b[0] + b[1] * x["T"] + b[2] * np.clip(x["T"] - s["map"], 0, None)
            bp = fit_line(x["T"], x["PE"], x["PEs"]["map"])
            fitp = bp[0] + bp[1] * x["T"] + bp[2] * np.clip(x["T"] - x["PEs"]["map"], 0, None)
            fn = os.path.join(a.csvdir, f"{a.csvprefix}_{x['lab'].replace('+', 'p')}_cooling_curve.csv")
            with open(fn, "w") as fh:
                fh.write(f"# {x['lab']}, {x['nat']} atoms, THOT {x['thot']:.0f} K; NPT quench segment, T <= {a.tmax:.0f} K, 2 ps averages\n")
                fh.write(f"# Tg(V) = {r['Tg_V']} K [95% CI {r['V_lo95']}-{r['V_hi95']}]; Tg(PE) = {r['Tg_PE']} K [{r['PE_lo95']}-{r['PE_hi95']}]\n")
                fh.write("T (K),V per atom (A^3),two-line fit V (A^3),PE per atom (eV),two-line fit PE (eV),density (g/cm3)\n")
                for k in range(len(x["T"])):
                    fh.write(f"{x['T'][k]:.2f},{x['V'][k]:.6f},{fit[k]:.6f},{x['PE'][k]:.6f},{fitp[k]:.6f},{x['rho'][k]:.5f}\n")
        fn = os.path.join(a.csvdir, f"{a.csvprefix}_posterior_p_Tg.csv")
        with open(fn, "w") as fh:
            fh.write("Tg (K)," + ",".join(f"p(Tg) from V {x['lab']},p(Tg) from PE {x['lab']}" for x in data) + "\n")
            for k, g in enumerate(grid):
                fh.write(f"{g:.2f}," + ",".join(f"{x['Vs']['post'][k]:.6g},{x['PEs']['post'][k]:.6g}" for x in data) + "\n")
        fn = os.path.join(a.csvdir, f"{a.csvprefix}_summary.csv")
        with open(fn, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) + (["Tg_exp_K"] if exp else []))
            w.writeheader()
            for r in rows:
                rr = dict(r)
                if exp:
                    rr["Tg_exp_K"] = next((v for k, v in exp.items() if r["run"].startswith(k)), "")
                w.writerow(rr)

    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"fit window T <= {a.tmax:.0f} K, Tg prior [{g0:.0f}, {g1:.0f}] K")
    print(f"{'run':24s} {'Tg(V) [95% CI]':>22s} {'Tg(PE) [95% CI]':>22s}  a_glass a_liq (1e-6/K)")
    for r in rows:
        print(f"{r['run']:24s} {r['Tg_V']:6d} [{r['V_lo95']:5d}-{r['V_hi95']:5d}]   {r['Tg_PE']:6d} [{r['PE_lo95']:5d}-{r['PE_hi95']:5d}]"
              f"   {r['alpha_glass_1e6']:6.1f} {r['alpha_liquid_1e6']:6.1f}")
    for j in joint:
        print(f"JOINT {j['group']:18s} n={j['n']} {j['prop']:2s}: Tg = {j['Tg']} K  [{j['lo95']}-{j['hi95']}]")


if __name__ == "__main__":
    main()
