#!/usr/bin/env python3
"""center_dump.py -- shift every frame of a LAMMPS text dump (id type element x y z) along one axis so that
the ZIF layer (circular mean of the Zn coordinates) sits in the middle of the box, then wrap all
coordinates into the box. Box, ids and order are unchanged.
  python center_dump.py in.lammpstrj out.lammpstrj [--axis z] [--el Zn]"""
import argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--axis", default="z"); ap.add_argument("--el", default="Zn")
a = ap.parse_args(); ax = "xyz".index(a.axis)
nfr = 0
with open(a.inp) as fi, open(a.out, "w") as fo:
    while True:
        hdr = [fi.readline() for _ in range(9)]
        if not hdr[0]:
            break
        n = int(hdr[3]); lo_hi = np.array([[float(v) for v in hdr[5 + d].split()[:2]] for d in range(3)])
        cols = hdr[8].split()[2:]; ix = [cols.index(c) for c in ("x", "y", "z")]; ie = cols.index("element")
        rows = [fi.readline().split() for _ in range(n)]
        xyz = np.array([[float(r[i]) for i in ix] for r in rows]); el = np.array([r[ie] for r in rows])
        lo, L = lo_hi[:, 0], lo_hi[:, 1] - lo_hi[:, 0]
        u = (xyz[el == a.el, ax] - lo[ax]) / L[ax]
        c = (np.angle(np.exp(2j * np.pi * u).mean()) / (2 * np.pi)) % 1.0
        xyz[:, ax] += (0.5 - c) * L[ax]
        xyz = lo + (xyz - lo) % L
        fo.writelines(hdr)
        for r, p in zip(rows, xyz):
            r[ix[0]], r[ix[1]], r[ix[2]] = f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"
            fo.write(" ".join(r) + "\n")
        nfr += 1
print(f"{a.out}: {nfr} frames")
