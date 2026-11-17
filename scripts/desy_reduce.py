#!/usr/bin/env python3
"""
desy_reduce.py -- ad hoc (PDFgetX3-style) reduction of DESY P02.1 pyFAI 1D patterns
(q.dat: Q [1/A], I) to S(Q), F(Q) = Q[S(Q)-1] and D(r) = 4 pi rho0 r [g(r)-1].

  I_s  = I_sample - alpha * I_empty
  y(Q) = Q * I_s / <f>^2                      (Cromer-Mann f, Faber-Ziman weights)
  F(Q) = [y(Q) - P_n(Q)] / N                  P_n: least-squares polynomial (degree n, default
                                              int(rpoly*Qmax/pi), rpoly 0.9) removes Compton,
                                              <f^2>/<f>^2 self scattering and slowly varying
                                              errors, exactly as in PDFgetX3
  N fixed by the low-r condition D(r) = -4 pi rho0 r for r < rmin (no pairs below rmin).
  D(r) = (2/pi) int Q [S-1] M(Q) sin(Qr) dQ, Lorch M -- identical to xrd_compare.py, so
  simulated and experimental D(r) use the same Q range and window.

  python desy_reduce.py --empty 133_empty...q.dat --out desy \
     --sample "30NaP:101_...q.dat:Na=0.6,P=1.4,O=3.8:2.40" \
     --sample "gZ62:104_...q.dat:Zn=1,C=7,H=6.5,N=4:1.56" ...
     (label:file:composition per formula unit or per 100 g:density g/cm3)
Writes <out>_SQ.csv, <out>_Dr.csv and <out>.npz.
"""
import argparse

import numpy as np
import periodictable as pt
import periodictable.cromermann as cm


def load(fn):
    d = np.loadtxt(fn, comments="#")
    return d[:, 0], d[:, 1]


def parse_comp(s):
    c = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        c[k] = float(v)
    return c


def comp_props(comp, dens):
    tot = sum(comp.values())
    x = {k: v / tot for k, v in comp.items()}
    mass = sum(x[k] * pt.elements.symbol(k).mass for k in x)          # g/mol per atom
    rho0 = dens / mass * 0.6022140857                                  # atoms / A^3
    return x, rho0


def fmoments(x, q):
    f = {k: cm.fxrayatq(k, q) for k in x}
    fav = sum(x[k] * f[k] for k in x)
    fsq = sum(x[k] * f[k] ** 2 for k in x)
    return fav, fsq


def sine_ft(q, iq, r, qmin, qmax, lorch=True):
    m = (q >= qmin) & (q <= qmax)
    q, iq = q[m], iq[m]
    M = np.sinc(q / qmax) if lorch else 1.0
    return 2 / np.pi * np.trapezoid(q * iq * M * np.sin(np.outer(r, q)), q, axis=1)


def compton(x, q):
    """approximate incoherent scattering per atom, I_inc = Z - f^2/Z (Q->0: 0, Q->inf: Z)"""
    return sum(x[k] * (pt.elements.symbol(k).number - cm.fxrayatq(k, q) ** 2 / pt.elements.symbol(k).number)
               for k in x)


def reduce(q, I, qe, Ie, x, rho0, alpha=1.0, qmin=0.8, qmax=20.0, rpoly=0.9, rmin=None):
    """I_s = aN(Q) [<f^2> + C + <f>^2 (S-1)];  aN(Q) = smooth polynomial fitted to I_s/(<f^2>+C)."""
    Ib = np.interp(q, qe, Ie)
    Is = I - alpha * Ib
    m = (q >= qmin) & (q <= qmax)
    q, Is = q[m], Is[m]
    fav, fsq = fmoments(x, q)
    base = fsq + compton(x, q)
    n = max(2, int(rpoly * qmax / np.pi))
    t = (q - q.mean()) / q.std()
    aN = np.polyval(np.polyfit(t, Is / base, n, w=q), t)          # weight by Q (F(Q) metric)
    S = 1 + (Is - aN * base) / (aN * fav ** 2)
    return q, S, q * (S - 1), aN.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--empty", required=True)
    ap.add_argument("--sample", action="append", required=True, help="label:file:comp:density")
    ap.add_argument("--alpha", type=float, default=1.0, help="empty-container scale")
    ap.add_argument("--qmin", type=float, default=0.8)
    ap.add_argument("--qmax", type=float, default=20.0)
    ap.add_argument("--rpoly", type=float, default=0.9)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qe, Ie = load(a.empty)
    r = np.arange(0.01, 12.0, 0.01)
    res = {}
    for spec in a.sample:
        lab, fn, comp, dens = spec.split(":")
        x, rho0 = comp_props(parse_comp(comp), float(dens))
        q, I = load(fn)
        qq, S, F, N = reduce(q, I, qe, Ie, x, rho0, a.alpha, a.qmin, a.qmax, a.rpoly)
        D = sine_ft(qq, S - 1, r, a.qmin, a.qmax)
        res[lab] = dict(q=qq, S=S, D=D, rho0=rho0, x=x)
        print(f"{lab:14s} rho0 {rho0:.4f} 1/A3  N {N:.3g}  S(Q) max {S.max():.2f} at {qq[S.argmax()]:.2f}")
    np.savez(a.out + ".npz", r=r, **{f"{k}__{f}": v[f] for k, v in res.items() for f in ("q", "S", "D")})
    labs = list(res)
    qq = res[labs[0]]["q"]
    with open(a.out + "_SQ.csv", "w") as f:
        f.write("Q," + ",".join(labs) + "\n")
        for k in range(len(qq)):
            f.write(f"{qq[k]:.4f}," + ",".join(f"{np.interp(qq[k], res[l]['q'], res[l]['S']):.5f}" for l in labs) + "\n")
    with open(a.out + "_Dr.csv", "w") as f:
        f.write("r," + ",".join(labs) + "\n")
        for k in range(len(r)):
            f.write(f"{r[k]:.3f}," + ",".join(f"{res[l]['D'][k]:.5f}" for l in labs) + "\n")


if __name__ == "__main__":
    main()
