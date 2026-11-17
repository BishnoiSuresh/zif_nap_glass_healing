"""common.py -- shared helpers for the post-processing scripts in post/.

A RUN directory is the folder in which run/run_step.sh was executed; it contains
composite_B/, heal_B_T<T>_s<seed>/, elastic_B/, zif221_B/, ...
"""
import glob
import os
import re
import sys

import numpy as np
from ase.data import atomic_numbers
from ase.io import read

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

EL = "C H N Na O P Zn".split()                     # composite type order (build_composite.py)
Z_OF_TYPE = {k + 1: atomic_numbers[e] for k, e in enumerate(EL)}
DT_PS = 0.001                                      # healing time step (ps per dump step)
AMU_A3_TO_G_CM3 = 1.66054


def read_data(fn, elements=EL):
    return read(fn, format="lammps-data", Z_of_type={k + 1: atomic_numbers[e] for k, e in enumerate(elements)},
                atom_style="atomic")


def heal_runs(run, suffix=""):
    """{(T, seed): folder} for every heal_B_T<T>_s<seed><suffix> in RUN"""
    out = {}
    for d in sorted(glob.glob(os.path.join(run, f"heal_B_T*_s*{suffix}"))):
        m = re.search(r"heal_B_T(\d+)_s(\d+)" + re.escape(suffix) + "$", d)
        if m and os.path.isdir(d):
            out[(int(m.group(1)), int(m.group(2)))] = d
    return out


def zif_centred_z(sym, z, Lz, el="Zn"):
    """z relative to the ZIF layer centre (circular mean of Zn along z), in (-Lz/2, Lz/2]"""
    z = np.asarray(z) % Lz
    zc = np.angle(np.exp(2j * np.pi * z[np.asarray(sym) == el] / Lz).mean()) / (2 * np.pi) * Lz
    return (z - zc + Lz / 2) % Lz - Lz / 2


def density_profile(sym, z, masses, L, dz=0.5):
    """ZIF-centred mass density profile along z (g/cm3); returns bin centres, rho"""
    u = zif_centred_z(sym, z, L[2])
    e = np.arange(-L[2] / 2, L[2] / 2 + dz, dz)
    h, _ = np.histogram(u, e, weights=masses)
    return 0.5 * (e[1:] + e[:-1]), h / (L[0] * L[1] * dz) * AMU_A3_TO_G_CM3


def void_width(sym, z, masses, L, dz=0.5, region=(-20.0, -3.0), rho_ref=1.5):
    """interfacial void width at the cracked interface I1 (below the ZIF centre), in A:
       sum over 0.5 A slices in `region` of dz * max(0, 1 - rho/rho_ref)."""
    c, rho = density_profile(sym, z, masses, L, dz)
    k = (c > region[0]) & (c < region[1])
    return dz * np.clip(1 - rho[k] / rho_ref, 0, None).sum(), rho[k].min()


MASS = {"C": 12.011, "H": 1.008, "N": 14.007, "Na": 22.990, "O": 15.999, "P": 30.974, "Zn": 65.38}


def iter_dump(fn):
    """yield (step, L[3], element array, xyz array) from a LAMMPS text dump with columns
       id type element x y z (as written by lammps/in.heal_r1 and in.composite_anneal)"""
    with open(fn) as fi:
        while True:
            hdr = [fi.readline() for _ in range(9)]
            if not hdr[0]:
                return
            step = int(hdr[1]); n = int(hdr[3])
            b = np.array([[float(v) for v in hdr[5 + d].split()[:2]] for d in range(3)])
            cols = hdr[8].split()[2:]
            ix = [cols.index(c) for c in ("x", "y", "z")]; ie = cols.index("element")
            rows = [fi.readline().split() for _ in range(n)]
            xyz = np.array([[float(r[i]) for i in ix] for r in rows]) - b[:, 0]
            yield step, b[:, 1] - b[:, 0], np.array([r[ie] for r in rows]), xyz


def mean_sd(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)
