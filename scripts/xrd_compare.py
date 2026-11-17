#!/usr/bin/env python3
"""
xrd_compare.py -- simulated X-ray structure factor and PDF vs experiment.

Simulated S(Q): computed DIRECTLY in reciprocal space from the periodic MD cells
(no g(r) truncation, valid down to Q_min = 2*pi/L):
    S_X(Q) - 1 = [ <|sum_j f_j(Q) exp(i k.r_j)|^2> / N  -  <f^2> ] / <f>^2     (Faber-Ziman)
averaged over k vectors with |k| in each Q bin (all vectors at low Q, a random
subset per bin at high Q) and over all frames of md_stats.lammpstrj (300 K NVT).
f_j(Q): Cromer-Mann (periodictable).  Compton/self terms are not part of i(Q).

Real space: the SAME sine transform is applied to experiment and simulation,
    D(r) = (2/pi) * int_{Qmin}^{Qmax} Q i(Q) M(Q) sin(Qr) dQ ,  M = Lorch,
so the Q range / resolution broadening is identical (D(r) = 4 pi rho0 r [G(r)-1]).
Partial g_ab(r) (r < L/2) are also written for interpretation.

  python xrd_compare.py --sys 40NaP --exp exp.int01 --dofr exp.dofr \
      --set "MH-1:runs/seed1/md_stats.lammpstrj,runs/seed2/...,..." \
      --set "MH-1+D3:runs/seed1_d3/md_stats.lammpstrj,..." --out xrd_40nap
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import periodictable.cromermann as cm

COLS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4"]
EXPC = "#0b0b0b"


def read_dump(fn, types=None):
    frames = []
    with open(fn) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: NUMBER OF ATOMS"):
            n = int(lines[i + 1])
            b = np.array([[float(x) for x in lines[i + 3 + d].split()[:2]] for d in range(3)])
            if len(lines[i + 3].split()) > 2:
                raise SystemExit("triclinic dump not supported")
            cols = lines[i + 6].split()[2:]
            rows = [l.split() for l in lines[i + 7:i + 7 + n]]
            if "element" in cols:
                el = np.array([r[cols.index("element")] for r in rows])
            else:
                el = np.array([types[int(r[cols.index("type")]) - 1] for r in rows])
            x = np.array([[float(r[cols.index(c)]) for c in ("x", "y", "z")] for r in rows])
            frames.append((el, x - b[:, 0], b[:, 1] - b[:, 0]))
            i += 7 + n
        else:
            i += 1
    return frames


def kvectors(L, qmax, dq, per_bin, rng, nfull=60000):
    """reciprocal-lattice vectors (half space): all of them up to Q_full, then per Q bin
    `per_bin` random directions rounded to the nearest lattice vector (binned by exact |k|)."""
    dk = 2 * np.pi / L
    qfull = min(qmax, (nfull * 3 / (2 * np.pi) * np.prod(dk)) ** (1 / 3))   # ~nfull vectors in half sphere
    nmax = np.ceil(qfull / dk).astype(int)
    g = [np.arange(-m, m + 1) for m in nmax]
    n = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    n = n[(n[:, 0] > 0) | ((n[:, 0] == 0) & (n[:, 1] > 0)) | ((n[:, 0] == 0) & (n[:, 1] == 0) & (n[:, 2] > 0))]
    k = n * dk
    q = np.linalg.norm(k, axis=1)
    k, q = k[q < qfull], q[q < qfull]
    if qfull < qmax:
        cen = np.arange(qfull + dq / 2, qmax, dq)
        u = rng.normal(size=(len(cen) * per_bin, 3))
        u /= np.linalg.norm(u, axis=1)[:, None]
        kk = np.round(u * np.repeat(cen, per_bin)[:, None] / dk) * dk
        kk = np.unique(kk, axis=0)
        qq = np.linalg.norm(kk, axis=1)
        m = (qq >= qfull) & (qq < qmax)
        k, q = np.vstack([k, kk[m]]), np.concatenate([q, qq[m]])
    b = (q / dq).astype(int)
    keep = []
    for bi in np.unique(b):
        idx = np.where(b == bi)[0]
        keep.append(idx if len(idx) <= per_bin else rng.choice(idx, per_bin, replace=False))
    keep = np.concatenate(keep)
    return k[keep], q[keep]


def sq_xray(frames, qmax=22.0, dq=0.05, per_bin=100, seed=0):
    rng = np.random.default_rng(seed)
    edges = np.arange(0, qmax + dq, dq)
    num = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1)
    for el, x, L in frames:
        k, q = kvectors(L, qmax, dq, per_bin, rng)
        bins = np.minimum((q / dq).astype(int), len(num) - 1)
        species = np.unique(el)
        N = len(el)
        for s0 in range(0, len(k), 2000):
            ks, qs = k[s0:s0 + 2000], q[s0:s0 + 2000]
            amp = np.zeros(len(ks), complex)
            fsq = np.zeros(len(ks))
            fav = np.zeros(len(ks))
            for sp in species:
                xs = x[el == sp]
                f = cm.fxrayatq(sp, qs)
                amp += f * np.exp(1j * ks @ xs.T).sum(1)
                c = len(xs) / N
                fsq += c * f ** 2
                fav += c * f
            iq = (np.abs(amp) ** 2 / N - fsq) / fav ** 2
            np.add.at(num, bins[s0:s0 + 2000], iq)
            np.add.at(cnt, bins[s0:s0 + 2000], 1)
    qc = 0.5 * (edges[1:] + edges[:-1])
    ok = cnt > 0
    return qc[ok], num[ok], cnt[ok]           # sum of i(Q) = S(Q) - 1 and number of k vectors per bin


def smooth_iq(qc, num, cnt, qout, sig_lo=0.12, sig_hi=0.04, qswitch=4.0):
    """count-weighted Gaussian smoothing (a small cell has only sparse k shells at low Q)."""
    out = np.full(len(qout), np.nan)
    for i, q0 in enumerate(qout):
        sg = sig_lo if q0 < qswitch else sig_hi
        w = cnt * np.exp(-0.5 * ((qc - q0) / sg) ** 2)
        if w.sum() > 1e-8 * cnt.max():
            out[i] = (w * num / np.maximum(cnt, 1)).sum() / w.sum()
    return out


def partial_gr(frames, pairs, dr=0.02):
    rmax = min(min(L) for _, _, L in frames) / 2
    edges = np.arange(0, rmax + dr, dr)
    out = {p: np.zeros(len(edges) - 1) for p in pairs}
    for el, x, L in frames:
        V = np.prod(L)
        for a, b in pairs:
            xa, xb = x[el == a], x[el == b]
            if len(xa) == 0 or len(xb) == 0:
                continue
            h = np.zeros(len(edges) - 1)
            for s in range(0, len(xa), 400):
                d = xa[s:s + 400, None, :] - xb[None, :, :]
                d -= L * np.round(d / L)
                r = np.sqrt((d ** 2).sum(-1)).ravel()
                h += np.histogram(r[(r > 1e-6) & (r < rmax)], edges)[0]
            shell = 4 / 3 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
            out[(a, b)] += h / (len(xa) * len(xb) / V * shell)
    rc = 0.5 * (edges[1:] + edges[:-1])
    keep = rc < rmax - 0.1                          # drop the half-box edge bins
    return rc[keep], {p: v[keep] / len(frames) for p, v in out.items()}


def sine_ft(q, iq, r, qmin, qmax):
    m = (q >= qmin) & (q <= qmax)
    q, iq = q[m], iq[m]
    M = np.sinc(q / qmax)                         # Lorch: sin(pi Q/Qmax)/(pi Q/Qmax)
    return 2 / np.pi * np.trapezoid(q * iq * M * np.sin(np.outer(r, q)), q, axis=1)


def read_int01(fn):
    d = np.loadtxt(fn, comments="#")
    return d[:, 0], d[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--exp", required=True, help="Gudrun .int01 i(Q) = S(Q)-1")
    ap.add_argument("--dofr", default=None, help="Gudrun .dofr (overlaid for reference)")
    ap.add_argument("--set", action="append", required=True, help="label:dump1,dump2,...")
    ap.add_argument("--pairs", default="", help="e.g. P-O,Na-O,O-O,P-P")
    ap.add_argument("--out", required=True)
    ap.add_argument("--qmax", type=float, default=None)
    ap.add_argument("--csvdir", default=None, help="write every plotted curve as Origin-ready CSV here")
    ap.add_argument("--csvprefix", default="", help="file-name prefix for --csvdir, e.g. FigS_MD1")
    a = ap.parse_args()

    qe, ie = read_int01(a.exp)
    good = ie != 0
    qe, ie = qe[good], ie[good]
    qmin, qmax = qe.min(), a.qmax or qe.max()
    r = np.arange(0.01, 12.0, 0.01)
    De = sine_ft(qe, ie, r, qmin, qmax)
    pairs = [tuple(p.split("-")) for p in a.pairs.split(",") if p]

    res = []
    for spec in a.set:
        lab, files = spec.split(":", 1)
        types = None
        if "|" in files:
            files, tl = files.split("|")
            types = tl.split()
        per_run = []
        allframes = []
        for fn in files.split(","):
            fr = read_dump(fn, types)
            allframes += fr
            import hashlib, os
            cache = "/tmp/xrdcache_" + hashlib.md5(os.path.abspath(fn).encode()).hexdigest() + ".npz"
            if os.path.exists(cache):
                z = np.load(cache); qs, nm, ct = z["qs"], z["nm"], z["ct"]
            else:
                qs, nm, ct = sq_xray(fr, qmax=max(qmax, 22.0))
                np.savez(cache, qs=qs, nm=nm, ct=ct)
            sm = smooth_iq(qs, nm, ct, qe)
            sm[qe < qs.min()] = np.nan
            per_run.append(sm)
            rho = np.mean([len(e) / np.prod(L) for e, _, L in fr])
        arr = np.array(per_run)
        mean = np.nanmean(arr, 0)
        Ds = np.array([sine_ft(qe, np.nan_to_num(x), r, qmin, qmax) for x in arr])
        rg, gp = partial_gr(allframes, pairs) if pairs else (None, None)
        qlow = 2 * np.pi / min(min(L) for _, _, L in allframes)
        m = qe >= max(qmin, qlow)
        rw = np.sqrt(np.sum((mean[m] - ie[m]) ** 2) / np.sum(ie[m] ** 2))
        res.append(dict(lab=lab, n=len(per_run), arr=arr, mean=mean, D=Ds, rg=rg, gp=gp, qlow=qlow, Rx=rw))
        print(f"{a.sys} {lab:22s} n={len(per_run)}  R_X(S(Q), Q>{max(qmin, qlow):.2f}) = {rw:.3f}")

    # ---- figure: S(Q) and D(r) --------------------------------------------
    fig, ax = plt.subplots(2, 1, figsize=(8.5, 8.2))
    ax[0].plot(qe, ie + 1, color=EXPC, lw=1.6, label="experiment (STOE, Gudrun)")
    for c, x in zip(COLS, res):
        lo, hi = np.nanmin(x["arr"], 0) + 1, np.nanmax(x["arr"], 0) + 1
        ax[0].fill_between(qe, lo, hi, color=c, alpha=0.18, lw=0)
        ax[0].plot(qe, x["mean"] + 1, color=c, lw=1.4,
                   label=f"{x['lab']} (n={x['n']}, R$_X$={x['Rx']:.2f})")
    ax[0].axhline(1, color="#c3c2b7", lw=0.8)
    ax[0].set_xlim(0, qmax)
    ax[0].set_xlabel("Q (Å$^{-1}$)")
    ax[0].set_ylabel("S$_X$(Q)")
    ax[0].set_title(f"{a.sys}: X-ray structure factor (Faber–Ziman)", fontsize=11)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(alpha=0.25, lw=0.6)
    ax[1].plot(r, De, color=EXPC, lw=1.6, label="experiment (same transform)")
    if a.dofr:
        d = np.loadtxt(a.dofr, comments="#")
        ax[1].plot(d[:, 0], d[:, 1], color="#8a8984", lw=1.0, ls="--", label="experiment (Gudrun .dofr)")
    for c, x in zip(COLS, res):
        ax[1].fill_between(r, x["D"].min(0), x["D"].max(0), color=c, alpha=0.18, lw=0)
        ax[1].plot(r, x["D"].mean(0), color=c, lw=1.4, label=x["lab"])
    ax[1].axhline(0, color="#c3c2b7", lw=0.8)
    ax[1].set_xlim(0.8, 10)
    ax[1].set_xlabel("r (Å)")
    ax[1].set_ylabel("D(r) (Å$^{-2}$)")
    ax[1].set_title(f"X-ray D(r) = 4πρ$_0$r[G(r)−1], Q = {qmin:.2f}–{qmax:.1f} Å$^{{-1}}$, Lorch", fontsize=11)
    ax[1].legend(fontsize=8, frameon=False)
    ax[1].grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(a.out + "_SQ_Dr.png", dpi=170)

    if pairs:
        ncol = min(4, len(pairs))
        nrow = int(np.ceil(len(pairs) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), squeeze=False)
        for axp, p in zip(axs.flat, pairs):
            for c, x in zip(COLS, res):
                axp.plot(x["rg"], x["gp"][p], color=c, lw=1.3, label=x["lab"])
            axp.set_title(f"g$_{{{p[0]}{p[1]}}}$(r)", fontsize=10)
            axp.set_xlim(0, 10)
            axp.set_xlabel("r (Å)")
            axp.grid(alpha=0.25, lw=0.6)
        axs.flat[0].legend(fontsize=7, frameon=False)
        for axp in list(axs.flat)[len(pairs):]:
            axp.axis("off")
        fig.tight_layout()
        fig.savefig(a.out + "_partials.png", dpi=170)

    with open(a.out + "_SQ.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Q", "S_exp"] + [f"S_{x['lab']}" for x in res])
        for k in range(len(qe)):
            w.writerow([f"{qe[k]:.4f}", f"{ie[k] + 1:.5f}"] + [f"{x['mean'][k] + 1:.5f}" for x in res])
    with open(a.out + "_Dr.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["r", "D_exp"] + [f"D_{x['lab']}" for x in res])
        for k in range(0, len(r), 2):
            w.writerow([f"{r[k]:.3f}", f"{De[k]:.5f}"] + [f"{x['D'].mean(0)[k]:.5f}" for x in res])

    if a.csvdir:
        write_csv(a, qe, ie, r, De, res, pairs)


def _clean(t):
    return (t.replace("Å", "A").replace("−", "-").replace("–", "-").replace(",", ";"))


def write_csv(a, qe, ie, r, De, res, pairs):
    """one CSV per panel; NaN -> empty cell (Origin reads it as missing)."""
    import os
    os.makedirs(a.csvdir, exist_ok=True)
    P = os.path.join(a.csvdir, a.csvprefix)

    def dump(fn, cols):
        names = [c[0] for c in cols]
        n = max(len(c[1]) for c in cols)
        with open(fn, "w") as fh:
            fh.write(",".join(_clean(x) for x in names) + "\n")
            for k in range(n):
                row = []
                for _, v in cols:
                    row.append("" if k >= len(v) or not np.isfinite(v[k]) else f"{v[k]:.6g}")
                fh.write(",".join(row) + "\n")

    cols = [("Q (1/A)", qe), ("S(Q) experiment", ie + 1)]
    for x in res:
        cols += [(f"S(Q) {x['lab']} mean", x["mean"] + 1),
                 (f"S(Q) {x['lab']} min over runs", np.nanmin(x["arr"], 0) + 1),
                 (f"S(Q) {x['lab']} max over runs", np.nanmax(x["arr"], 0) + 1)]
    dump(P + "_A_SQ.csv", cols)
    cols = [("r (A)", r), ("D(r) experiment (same transform; 1/A^2)", De)]
    if a.dofr:
        d = np.loadtxt(a.dofr, comments="#")
        cols += [("r Gudrun dofr (A)", d[:, 0]), ("D(r) experiment Gudrun dofr", d[:, 1])]
    for x in res:
        cols += [(f"D(r) {x['lab']} mean", x["D"].mean(0)),
                 (f"D(r) {x['lab']} min over runs", x["D"].min(0)),
                 (f"D(r) {x['lab']} max over runs", x["D"].max(0))]
    dump(P + "_A_Dr.csv", cols)
    if pairs:
        cols = []
        for x in res:
            cols.append((f"r {x['lab']} (A)", x["rg"]))
            cols += [(f"g_{p[0]}{p[1]}(r) {x['lab']}", x["gp"][p]) for p in pairs]
        dump(P + "_B_partial_gr.csv", cols)


if __name__ == "__main__":
    main()
