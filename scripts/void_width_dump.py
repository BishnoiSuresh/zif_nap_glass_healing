#!/usr/bin/env python3
"""void_width_dump.py -- interfacial void width at the cracked interface I1 for every frame of a LAMMPS text dump
(id type element x y z): ZIF-centred mass-density profile along z (0.5 A bins); void width =
sum dz*max(0, 1 - rho/1.5 g/cm3) over z - z_ZIF in (-20, -3) A.  Prints step, Lz, void width.
  python void_width_dump.py dump.lammpstrj [--from STEP]"""
import argparse
import numpy as np
M = {"C": 12.011, "H": 1.008, "N": 14.007, "Na": 22.990, "O": 15.999, "P": 30.974, "Zn": 65.38}
ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--from", dest="frm", type=int, default=-1)
a = ap.parse_args()
with open(a.dump) as fi:
    while True:
        hdr = [fi.readline() for _ in range(9)]
        if not hdr[0]:
            break
        step = int(hdr[1]); n = int(hdr[3])
        b = np.array([[float(v) for v in hdr[5 + d].split()[:2]] for d in range(3)]); L = b[:, 1] - b[:, 0]
        cols = hdr[8].split()[2:]; iz = cols.index("z"); ie = cols.index("element")
        rows = [fi.readline().split() for _ in range(n)]
        if step < a.frm:
            continue
        z = np.array([float(r[iz]) for r in rows]) - b[2, 0]; el = np.array([r[ie] for r in rows]); m = np.array([M[e] for e in el])
        u = (z % L[2]) / L[2]; zc = (np.angle(np.exp(2j * np.pi * u[el == "Zn"]).mean()) / (2 * np.pi)) % 1 * L[2]
        w = (z - zc + L[2] / 2) % L[2] - L[2] / 2
        dz = 0.5; e = np.arange(-L[2] / 2, L[2] / 2 + dz, dz); h, _ = np.histogram(w, e, weights=m)
        rho = h / (L[0] * L[1] * dz) * 1.66054; c = 0.5 * (e[1:] + e[:-1]); k = (c > -20) & (c < -3)
        print(step, round(L[2], 3), round(dz * np.clip(1 - rho[k] / 1.5, 0, None).sum(), 3))
