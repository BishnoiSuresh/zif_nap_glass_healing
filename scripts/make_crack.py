#!/usr/bin/env python3
"""
make_crack.py -- open a planar through-crack in a periodic composite glass.

A plane normal to --axis (default x, i.e. perpendicular to the NaP|ZIF layers that
are stacked along z, so the crack cuts both phases and both interfaces) is placed
at --pos (fraction of the box).  Chemical units are kept whole and moved rigidly:
  * ZIF linkers  (C/N/H connected components)  -> by centre of mass
  * PO4 groups   (P + its O, O assigned to the nearest P) -> by P position
  * Zn, Na, isolated atoms -> by their own position
Units whose centre lies on the + side of the plane are shifted by --gap and the box
grows by --gap, so the bonds that cross the plane (Zn-N, P-O-P, Na-O, ...) are
broken and two free surfaces separated by ~gap face each other.  Periodic images are
untouched (only ONE crack per cell).

  python make_crack.py composite_glass.data --elements C H N Na O P Zn --gap 5 --out cracked.data
Writes cracked.data (LAMMPS, same type order) + crack_info.txt (broken bonds by type).

Interfacial mode -- crack ALONG a ZIF|NaP interface (plane parallel to the layers):
  python make_crack.py composite_glass.data --elements C H N Na O P Zn --axis z --plane auto \
         --interface 1 --gap 5 --out cracked.data
  --plane auto scans planes normal to --axis and puts the crack where it cuts the most Zn-O-P
  bonds (Zn and P on opposite sides), at interface 1 (below the ZIF centre) or 2 (above).
  Ties are broken by the fewest P-O-P and Zn-N bonds cut. The other interface stays intact
  and is the built-in control. crack_meta.json records both interface planes.
"""
import json
import argparse
from collections import Counter

import numpy as np
from ase.data import atomic_numbers
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

LINK = {("C", "C"): 1.75, ("C", "N"): 1.75, ("N", "N"): 1.6, ("C", "H"): 1.25, ("N", "H"): 1.2}
BONDS = {("Zn", "N"): 2.5, ("Zn", "O"): 2.5, ("P", "O"): 2.05, ("Na", "O"): 3.0, ("Na", "N"): 3.0}


def auto_plane(at, s, unit, cen, L, ax, which, dz=0.1):
    """Plane normal to ax cutting the most Zn-O-P bonds at interface `which` (1 below, 2 above the ZIF
    centre). A bond is cut when the centres of its two units lie on opposite sides of the plane."""
    Lx = L[ax]
    pos = at.get_positions() % L
    i, j, D = neighbor_list("ijD", at, {("Zn", "O"): 2.5, ("P", "O"): 2.05, ("Zn", "N"): 2.5})
    ob = set(j[(s[i] == "P") & (s[j] == "O")].tolist())
    zn = s == "Zn"
    zc = np.angle(np.exp(2j * np.pi * pos[zn, ax] / Lx).mean()) / (2 * np.pi) * Lx % Lx

    def seg(a_, b_):                       # unit-centre segment (start, signed length) along ax, min image
        ca, cb = cen[unit[a_], ax], cen[unit[b_], ax]
        d = cb - ca
        d -= Lx * np.round(d / Lx)
        return ca, d

    sets = {"ZnOP": [], "POP": [], "ZnN": []}
    for p, q in zip(i, j):
        if s[p] == "Zn" and s[q] == "O" and q in ob:
            sets["ZnOP"].append(seg(p, q))
        elif s[p] == "Zn" and s[q] == "N":
            sets["ZnN"].append(seg(p, q))
        elif s[p] == "P" and s[q] == "O" and unit[p] != unit[q]:
            sets["POP"].append(seg(p, q))     # O belongs to another P's PO4 -> bridge crossing units
    grid = np.arange(0, Lx, dz)

    def count(lst):
        if not lst:
            return np.zeros(len(grid))
        st = np.array([x[0] for x in lst])
        ln = np.array([x[1] for x in lst])
        lo = np.minimum(st, st + ln)
        hi = np.maximum(st, st + ln)
        g = grid[:, None]
        hit = ((g > lo) & (g < hi)) | ((g + Lx > lo) & (g + Lx < hi)) | ((g - Lx > lo) & (g - Lx < hi))
        return hit.sum(1)

    nZ, nP, nN = count(sets["ZnOP"]), count(sets["POP"]), count(sets["ZnN"])
    u = (grid - zc + Lx / 2) % Lx - Lx / 2         # position relative to the ZIF centre
    best = {}
    for k, msk in ((1, u < 0), (2, u >= 0)):
        sc = np.where(msk, nZ * 1000 - (nP + nN), -1e9)
        b = int(np.argmax(sc))
        best[k] = dict(frac=float(grid[b] / Lx), ZnOP=int(nZ[b]), POP=int(nP[b]), ZnN=int(nN[b]))
    other = 2 if which == 1 else 1
    if best[which]["ZnOP"] == 0:
        print("WARNING: no Zn-O-P bond found at the requested interface -- is the composite annealed?")
    return best[which]["frac"], dict(zif_centre_frac=float(zc / Lx), other_plane_frac=best[other]["frac"],
                                     scores={k: best[which][k] for k in ("ZnOP", "POP", "ZnN")},
                                     other_scores={k: best[other][k] for k in ("ZnOP", "POP", "ZnN")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--elements", nargs="+", required=True)
    ap.add_argument("--axis", default="x", choices=["x", "y", "z"])
    ap.add_argument("--pos", type=float, default=0.5, help="crack plane, fraction of box along axis")
    ap.add_argument("--plane", default="pos", choices=["pos", "auto"],
                    help="pos = use --pos; auto = interfacial crack at the plane cutting most Zn-O-P bonds")
    ap.add_argument("--interface", type=int, default=1, choices=[1, 2],
                    help="with --plane auto: 1 = interface below the ZIF centre, 2 = above")
    ap.add_argument("--gap", type=float, default=5.0, help="crack opening (A)")
    ap.add_argument("--out", default="cracked.data")
    a = ap.parse_args()
    ax = "xyz".index(a.axis)

    Z = {k + 1: atomic_numbers[e] for k, e in enumerate(a.elements)}
    at = read(a.data, format="lammps-data", Z_of_type=Z, atom_style="atomic")
    cell = np.array(at.cell)
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-6:
        raise SystemExit("orthogonal cell expected")
    L = np.diag(cell).copy()
    s = np.array(at.get_chemical_symbols())
    pos = at.get_positions() % L
    n = len(at)

    # --- units ---------------------------------------------------------
    unit = -np.ones(n, int)
    nu = 0
    org = np.where(np.isin(s, ["C", "N", "H"]))[0]
    if len(org):
        sub = at[org]
        i, j = neighbor_list("ij", sub, LINK)
        _, lab = connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(len(org),) * 2))
        unit[org] = lab
        nu = lab.max() + 1
    P = np.where(s == "P")[0]
    if len(P):
        unit[P] = nu + np.arange(len(P))
        O = np.where(s == "O")[0]
        d = pos[O][:, None, :] - pos[P][None, :, :]
        d -= L * np.round(d / L)
        unit[O] = nu + np.argmin((d ** 2).sum(-1), axis=1)
        nu += len(P)
    rest = np.where(unit < 0)[0]
    unit[rest] = nu + np.arange(len(rest))
    nu += len(rest)

    # unwrap each unit around its anchor, centre = mean (P position for PO4)
    new = pos.copy()
    cen = np.zeros((nu, 3))
    for u in range(nu):
        idx = np.where(unit == u)[0]
        anchor = idx[0]
        if len(P) and s[idx].tolist().count("P") == 1:
            anchor = idx[s[idx] == "P"][0]
        d = pos[idx] - pos[anchor]
        d -= L * np.round(d / L)
        new[idx] = pos[anchor] + d
        c = pos[anchor] if s[anchor] == "P" else new[idx].mean(0)
        # bring the whole unit into the image whose centre lies inside the box, so that the
        # side assignment (from the centre) and the atom positions refer to the SAME image
        img = np.floor(c / L)
        new[idx] -= img * L
        cen[u] = c - img * L

    zinfo = {}
    if a.plane == "auto":
        a.pos, zinfo = auto_plane(at, s, unit, cen, L, ax, a.interface)
        print(f"auto plane: interface {a.interface} at {a.pos * L[ax]:.2f} A ({a.pos:.4f} of box); "
              f"other interface at {zinfo['other_plane_frac'] * L[ax]:.2f} A; scores {zinfo['scores']}")

    # broken bonds = bonded pairs whose units end up on different sides
    xc = a.pos * L[ax]
    side = cen[:, ax] >= xc
    i, j, D = neighbor_list("ijD", at, BONDS)
    cross = side[unit[i]] != side[unit[j]]
    # broken = the two UNITS end up on different sides and the segment between the unit centres
    # crosses the crack plane (not the periodic boundary, where nothing is opened)
    ci = cen[unit[i], ax]
    dc = cen[unit[j], ax] - ci
    dc -= L[ax] * np.round(dc / L[ax])
    spans = ((ci - xc) * (ci + dc - xc)) < 0
    br = Counter("-".join(sorted((s[p], s[q]))) for p, q in zip(i[cross & spans], j[cross & spans]) if p < q)

    shift = np.zeros(n)
    shift[side[unit]] = a.gap
    new[:, ax] += shift
    L2 = L.copy()
    L2[ax] += a.gap
    at.set_cell(np.diag(L2))
    at.set_positions(new % L2)
    write(a.out, at, format="lammps-data", specorder=a.elements, atom_style="atomic", masses=True)
    # per-atom side (0 = fixed side, 1 = shifted side) + plane, for heal_analysis.py
    np.savetxt("crack_side.txt", side[unit].astype(int), fmt="%d",
               header=f"axis {ax} plane_frac_pristine {a.pos} plane_frac_cracked {(xc + a.gap / 2) / L2[ax]:.6f}")
    with open("crack_info.txt", "w") as f:
        f.write(f"crack plane {a.axis} = {xc:.3f} A ({a.pos:.2f} of box), opening {a.gap} A\n")
        f.write(f"box {np.round(L, 3)} -> {np.round(L2, 3)}\n")
        f.write(f"units moved: {int(side.sum())} of {nu}\n")
        f.write(f"bonds broken across the crack plane: {dict(br)}\n")
        if zinfo:
            f.write(f"interfacial crack at interface {a.interface}; Zn-O-P bonds cut: {zinfo['scores']['ZnOP']}; "
                    f"other (control) interface at {zinfo['other_plane_frac']:.4f} of the pristine box\n")
    other_p = zinfo.get("other_plane_frac")
    meta = dict(axis=int(ax), gap=a.gap, plane_frac_pristine=a.pos, plane_frac_cracked=(xc + a.gap / 2) / L2[ax],
                L_pristine=float(L[ax]), L_cracked=float(L2[ax]), mode=a.plane, interface=a.interface if zinfo else None,
                zif_centre_frac=zinfo.get("zif_centre_frac"),
                other_plane_frac_pristine=other_p,
                other_plane_frac_cracked=None if other_p is None else
                ((other_p * L[ax] + (a.gap if other_p * L[ax] >= xc else 0.0)) / L2[ax]),
                broken=dict(br), cut_scores=zinfo.get("scores"))
    json.dump(meta, open("crack_meta.json", "w"), indent=1)
    print(open("crack_info.txt").read())


if __name__ == "__main__":
    main()
