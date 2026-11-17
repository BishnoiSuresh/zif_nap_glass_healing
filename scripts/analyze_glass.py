#!/usr/bin/env python3
"""
analyze_glass.py  --  quick structural check of a LAMMPS glass (any of the 3 systems)
====================================================================================
  python analyze_glass.py runs/40nap/seed1/glass_300K.dat --elements Na O P
  python analyze_glass.py runs/zif62/seed*/glass_300K.dat  --elements C H N Zn
  python analyze_glass.py runs/comp/seed*/glass_300K.dat   --elements C H N Na O P Zn --csv summary.csv

Reports per file: density; ZIF linker integrity (intact Im / bIm, broken pieces);
Zn-N / Zn-O CN and % Zn with >=1 O; P-O CN distribution; Q^n; O2 and P-P defects;
Na-O CN.  Several files -> also mean +- std over replicas.
"""
import argparse
import csv
from collections import Counter

import numpy as np
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CUT = {("C", "C"): 1.75, ("C", "N"): 1.75, ("C", "H"): 1.25, ("N", "H"): 1.20}
PO, ZN, NAO = 2.05, 2.5, 3.0


def formula(sym):
    c = Counter(sym)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def analyse(at):
    s = np.array(at.get_chemical_symbols())
    r = {"atoms": len(at), "rho": at.get_masses().sum() * 1.66054 / at.get_volume()}
    if np.isin(s, ["C", "N", "H"]).any():
        sub = at[np.where(np.isin(s, ["C", "N", "H"]))[0]]
        i, j = neighbor_list("ij", sub, CUT)
        _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(sub),) * 2))
        ss = np.array(sub.get_chemical_symbols())
        c = Counter(formula(ss[lab == k]) for k in np.unique(lab))
        r["Im"], r["bIm"] = c.pop("C3H3N2", 0), c.pop("C7H5N2", 0)
        r["broken"] = sum(c.values())
    if "Zn" in s:
        i, j = neighbor_list("ij", at, {("Zn", "N"): ZN, ("Zn", "O"): ZN})
        zn = np.where(s == "Zn")[0]
        nN = np.array([((i == z) & (s[j] == "N")).sum() for z in zn])
        nO = np.array([((i == z) & (s[j] == "O")).sum() for z in zn])
        r["ZnN"], r["ZnO"], r["Zn_with_O_%"] = nN.mean(), nO.mean(), 100 * (nO > 0).mean()
    if "P" in s:
        i, j = neighbor_list("ij", at, {("P", "O"): PO})
        m = s[i] == "P"
        cnP = np.bincount(i[m], minlength=len(at))[s == "P"]
        r["P_4coord_%"] = 100 * (cnP == 4).mean()
        oP = np.bincount(j[m], minlength=len(at))
        bo = set(np.where((s == "O") & (oP == 2))[0])
        q = Counter(sum(1 for o in j[m & (i == p)] if o in bo) for p in np.where(s == "P")[0])
        n = (s == "P").sum()
        for k in range(5):
            r[f"Q{k}_%"] = 100 * q.get(k, 0) / n
        i2, j2 = neighbor_list("ij", at, {("O", "O"): 1.6, ("P", "P"): 2.5})
        r["O2"] = int(((s[i2] == "O") & (s[j2] == "O")).sum() // 2)
        r["P-P"] = int(((s[i2] == "P") & (s[j2] == "P")).sum() // 2)
    if "Na" in s:
        i, j = neighbor_list("ij", at, {("Na", "O"): NAO})
        r["NaO"] = (s[i] == "Na").sum() / (s == "Na").sum()
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--elements", nargs="+", required=True, help="LAMMPS type order")
    p.add_argument("--csv", default=None)
    a = p.parse_args()
    Z = {k + 1: atomic_numbers[e] for k, e in enumerate(a.elements)}
    rows = []
    for f in a.files:
        at = read(f, format="lammps-data", Z_of_type=Z, atom_style="atomic") if f.endswith(".dat") or f.endswith(".data") else read(f)
        r = analyse(at)
        r["file"] = f
        rows.append(r)
        print(f, {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in r.items() if k != "file"})
    if len(rows) > 1:
        keys = [k for k in rows[0] if k != "file" and isinstance(rows[0][k], (int, float))]
        print("\nmean +- std over", len(rows), "replicas")
        for k in keys:
            v = np.array([r.get(k, np.nan) for r in rows], float)
            print(f"  {k:14s} {np.nanmean(v):9.3f} +- {np.nanstd(v, ddof=1) if len(v) > 1 else 0:7.3f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
