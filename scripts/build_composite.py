#!/usr/bin/env python3
"""
build_composite.py  (v3 - NaP glass kept EXACT, ZIF linkers kept rigid)
=======================================================================
Build a layered ZIF-62 / 40NaP composite starting structure from the two
validated melt-quenched glasses.

    z ^   +-----------------------+  <- periodic boundary (interface 2)
      |   |   ZIF-62 glass layer  |     linkers rigid, Zn + linker COMs remapped
      |   +-----------------------+  <- interface 1
      |   |                       |
      |   |   40NaP glass         |     ALL coordinates = original glass
      |   |   (untouched)         |     (only a rigid translation along c)
      |   +-----------------------+
          lateral cell = NaP a, b vectors (exactly)

NaP (v3 change)
---------------
* The composite lateral vectors a, b are the NaP cell vectors themselves,
  and the NaP c vector is along z, so every NaP atom keeps its exact
  Cartesian coordinates (rigid translation only).  All P-O bonds, P-O-P
  bridges, Na sites and angles are identical to the validated glass,
  except the P-O-P bridges that cross the ONE cut plane that opens the
  periodic NaP cell along z.
* The cut plane is chosen by scanning the whole cell for the position
  that breaks the fewest P-O-P bridges.  Each broken bridge necessarily
  leaves one 3-coordinated P at the interface (O/P = 2.83 is fixed by
  stoichiometry); these reconnect during relaxation/annealing.
* No PO4 is ever split: every O is placed at P + (original P-O vector).

ZIF
---
* Units: Im (C3H3N2) and bIm (C7H5N2) linkers (rigid, whole) and Zn ions.
* Its own cut plane is chosen to break the fewest Zn-N bonds.
* Unit centres are mapped from the ZIF cell into the NaP lateral cell with
  the ZIF volume preserved (h = V_zif / A_lateral); linkers are rigidly
  translated, so ring geometry is identical to the ZIF glass.  Zn-N
  distances are strained by this lateral mismatch and re-form on annealing.
* Clashes (ZIF-ZIF and ZIF-NaP) are removed by rigid moves of ZIF units
  only; the NaP layer is never moved.

Box
---
  Lz = c_NaP + h_ZIF + 2*gap      (gap default 0.5 A per interface)
so the starting density is close to the ideal two-phase value.

Usage
-----
  python build_composite.py \
      --zif zif62_glass_final.xyz --nap 40nap_glass_final.xyz \
      --n_zif 1 --n_nap 1 --output composite_1zif_1nap.xyz --seed 42

  # optional (recommended) pre-step: make the ZIF glass fit the NaP lateral cell
  python build_composite.py --zif zif62_glass_final.xyz --nap 40nap_glass_final.xyz \
      --write_zif_reshape            # -> zif62_reshape_start.data + in.zif62_reshape
  (run in.zif62_reshape with LAMMPS/MACE, then build with --zif zif62_reshape_final.data)

  python build_composite.py --check composite_glass_final.xyz   # integrity only
  python build_composite.py --check zif62_reshape_final.data     # (types C H N Zn)
"""
import argparse
import os
from collections import Counter

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, atomic_numbers
from ase.io import read, write
from ase.neighborlist import neighbor_list
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ZIF_CUT = {("C", "C"): 1.75, ("C", "N"): 1.75, ("C", "H"): 1.25, ("N", "H"): 1.20}
PO_CUT = 2.05
ZNN_CUT = 2.50
LINKERS = {"C3H3N2": "Im", "C7H5N2": "bIm"}
SPECORDER = ["C", "H", "N", "Na", "O", "P", "Zn"]
ZN_MIN = 1.85  # min Zn-N / Zn-O distance between different units (A)


# ----------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------
def mass_of(symbols):
    return float(sum(atomic_masses[atomic_numbers[s]] for s in symbols))


def density_gcc(atoms):
    return mass_of(atoms.get_chemical_symbols()) * 1.66054 / atoms.get_volume()


def formula(symbols):
    c = Counter(symbols)
    return "".join(f"{e}{c[e]}" for e in sorted(c))


def components(atoms, cutoff):
    n = len(atoms)
    i, j, D = neighbor_list("ijD", atoms, cutoff)
    g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab, i, j, D


def unwrap_units(atoms, lab, i, j, D):
    """Make every covalent unit whole (BFS with minimum-image bond vectors)."""
    pos = atoms.get_positions()
    new = np.full_like(pos, np.nan)
    nbrs = [[] for _ in range(len(atoms))]
    for a, b, d in zip(i, j, D):
        nbrs[a].append((b, d))
    for root in np.unique(lab):
        s = np.where(lab == root)[0][0]
        new[s] = pos[s]
        stack = [s]
        while stack:
            a = stack.pop()
            for b, d in nbrs[a]:
                if np.isnan(new[b, 0]):
                    new[b] = new[a] + d
                    stack.append(b)
    return new


def integrity_report(atoms, label=""):
    sym = np.array(atoms.get_chemical_symbols())
    out = {}
    if np.isin(sym, ["C", "N", "H"]).any():
        sub = atoms[np.where(np.isin(sym, ["C", "N", "H"]))[0]]
        lab, *_ = components(sub, ZIF_CUT)
        ss = np.array(sub.get_chemical_symbols())
        c = Counter(formula(ss[lab == k]) for k in np.unique(lab))
        out["Im"] = c.pop("C3H3N2", 0)
        out["bIm"] = c.pop("C7H5N2", 0)
        out["broken_CNH_fragments"] = sum(c.values())
        if c:
            out["fragment_types"] = dict(c.most_common(6))
    if "Zn" in sym:
        i, j = neighbor_list("ij", atoms, {("Zn", "N"): ZNN_CUT})
        cn = np.bincount(i[sym[i] == "Zn"], minlength=len(atoms))[sym == "Zn"]
        out["Zn_CN(N<2.5)"] = dict(sorted(Counter(cn.tolist()).items()))
    if "P" in sym:
        i, j = neighbor_list("ij", atoms, {("P", "O"): PO_CUT})
        cn = np.bincount(i[sym[i] == "P"], minlength=len(atoms))[sym == "P"]
        out["P_CN(O)"] = dict(sorted(Counter(cn.tolist()).items()))
    if "O" in sym:
        i, j = neighbor_list("ij", atoms, {("O", "O"): 1.6})
        out["O-O<1.6A (O2)"] = int(len(i) // 2)
        if "P" in sym:
            i, j = neighbor_list("ij", atoms, {("O", "P"): PO_CUT})
            out["O_without_P"] = int((np.bincount(i, minlength=len(atoms))[sym == "O"] == 0).sum())
    print(f"  Integrity {label}: {out}")
    return out


def min_contacts(atoms, pairs=(("Zn", "N"), ("Zn", "O"), ("Na", "O"), ("P", "O"),
                               ("H", "H"), ("C", "O"), ("N", "O"), ("H", "O"), ("C", "C"))):
    sym = np.array(atoms.get_chemical_symbols())
    i, j, d = neighbor_list("ijd", atoms, 3.0)
    res = {}
    for a, b in pairs:
        m = (sym[i] == a) & (sym[j] == b)
        if m.any():
            res[f"{a}-{b}"] = round(float(d[m].min()), 3)
    return res


def to_standard(atoms):
    """Rigid rotation to LAMMPS-style lower-triangular cell (a || x, b in xy).
    Distances and angles are unchanged (pure rotation)."""
    atoms = atoms.copy()
    rcell, _ = atoms.cell.standard_form()
    atoms.set_cell(rcell, scale_atoms=True)
    return atoms


def read_structure(path, lammps_types):
    if path.endswith((".data", ".dat", ".lmp")):
        Z = {k + 1: atomic_numbers[e] for k, e in enumerate(lammps_types)}
        return read(path, format="lammps-data", Z_of_type=Z, atom_style="atomic")
    return read(path)


def frac_z(cell, vec):
    """Fractional component along c of Cartesian vector(s)."""
    return np.linalg.solve(np.array(cell).T, np.atleast_2d(vec).T)[2]


# ----------------------------------------------------------------------
# NaP: exact slab, best cut plane
# ----------------------------------------------------------------------
def nap_bridges(nap):
    """P-O bonds and P-O-P bridges (with P1->P2 vector through the O)."""
    sym = np.array(nap.get_chemical_symbols())
    i, j, D = neighbor_list("ijD", nap, {("P", "O"): PO_CUT})
    m = sym[i] == "P"
    bonds = {}                                   # O -> [(P, vec P->O), ...]
    for p, o, d in zip(i[m], j[m], D[m]):
        bonds.setdefault(o, []).append((p, d))
    bridges = [(v[0][0], v[1][0], v[0][1] - v[1][1]) for v in bonds.values() if len(v) == 2]
    return bonds, bridges


def count_broken(fz_units, links, z0):
    """links: (u1, u2, dfz). Broken if u2 is not in the same periodic slab as u1."""
    n = 0
    for u1, u2, dfz in links:
        z1 = (fz_units[u1] - z0) % 1.0
        z2 = z1 + dfz
        if z2 < 0.0 or z2 >= 1.0:
            n += 1
    return n


def best_cut(fz_units, links, n_scan=400):
    scan = [(count_broken(fz_units, links, z0), z0)
            for z0 in np.linspace(0, 1, n_scan, endpoint=False)]
    worst = max(scan)[0]
    return min(scan)[1], min(scan)[0], worst


def build_nap_slab(nap):
    cell = np.array(nap.cell)
    if abs(cell[0, 2]) > 1e-8 or abs(cell[1, 2]) > 1e-8:
        raise SystemExit("NaP a, b must lie in the xy plane (call to_standard first).")
    sym = np.array(nap.get_chemical_symbols())
    pos0 = nap.get_positions()
    frac = nap.get_scaled_positions(wrap=True)
    bonds, bridges = nap_bridges(nap)
    links = [(p1, p2, frac_z(cell, v)[0]) for p1, p2, v in bridges]
    z0, nbroken, worst = best_cut(frac[:, 2], links)
    print(f"  NaP: {len(bridges)} P-O-P bridges; best cut at c-frac {z0:.3f} breaks "
          f"{nbroken} (worst plane would break {worst})")

    fnew = frac.copy()
    fnew[:, 2] = (frac[:, 2] - z0) % 1.0
    pos = fnew @ cell                      # rigid translation (+ lattice vectors)
    # every O sits at P + original P-O vector -> no PO4 is split
    for o, plist in bonds.items():
        if len(plist) == 1:
            p, d = plist[0]
        else:  # bridge: choose the P that keeps O inside the slab
            best = None
            for p, d in plist:
                z = pos[p, 2] + d[2]
                pen = max(0.0, -z) + max(0.0, z - cell[2, 2])
                if best is None or pen < best[0]:
                    best = (pen, p, d)
            _, p, d = best
        pos[o] = pos[p] + d
    # verify exactness: new - old must be (translation) + lattice vectors
    t = -z0 * cell[2]
    dev = np.linalg.solve(cell.T, (pos - pos0 - t).T).T
    err = np.abs(dev - np.round(dev)) @ np.abs(cell)
    print(f"  NaP coordinates preserved: max deviation from rigid translation "
          f"= {err.max():.2e} A")
    return sym, pos, cell[2, 2], nbroken


# ----------------------------------------------------------------------
# ZIF: rigid linkers, best cut plane, remap into NaP lateral cell
# ----------------------------------------------------------------------
def build_zif_layer(zif, a_vec, b_vec, z_start):
    cell = np.array(zif.cell)
    sym = np.array(zif.get_chemical_symbols())
    lab, i, j, D = components(zif, ZIF_CUT)
    whole = unwrap_units(zif, lab, i, j, D)
    masses = zif.get_masses()
    units, kinds, coms = [], [], []
    for k in np.unique(lab):
        idx = np.where(lab == k)[0]
        f = formula(sym[idx])
        kind = "Zn" if (len(idx) == 1 and sym[idx[0]] == "Zn") else LINKERS.get(f, "BROKEN:" + f)
        units.append(idx)
        kinds.append(kind)
        coms.append((whole[idx] * masses[idx, None]).sum(0) / masses[idx].sum())
    bad = [k for k in kinds if k.startswith("BROKEN")]
    if bad:
        print(f"  WARNING: input ZIF glass has broken units: {Counter(bad)}")
    coms = np.array(coms)
    unit_of = np.empty(len(zif), int)
    for u, idx in enumerate(units):
        unit_of[idx] = u
    fcom = np.linalg.solve(cell.T, coms.T).T % 1.0

    # Zn-linker links (Zn unit -> linker unit, vector Zn -> linker COM, min image)
    ii, jj, DD = neighbor_list("ijD", zif, {("Zn", "N"): ZNN_CUT})
    links = []
    for zn, n, d in zip(ii, jj, DD):
        if sym[zn] != "Zn" or sym[n] != "N":
            continue
        uz, ul = unit_of[zn], unit_of[n]
        v = d + (coms[ul] - whole[n])          # Zn -> N -> linker COM
        links.append((uz, ul, frac_z(cell, v)[0]))
    z0, nbroken, worst = best_cut(fcom[:, 2], links)
    print(f"  ZIF: {len(links)} Zn-N bonds; best cut at c-frac {z0:.3f} breaks {nbroken} "
          f"(worst plane would break {worst})")

    area = np.linalg.norm(np.cross(a_vec, b_vec))
    h = zif.get_volume() / area
    c_new = cell[2] * (h / cell[2, 2])            # keep the ZIF c tilt, rescale height
    fz = (fcom[:, 2] - z0) % 1.0
    new_com = fcom[:, 0, None] * a_vec + fcom[:, 1, None] * b_vec + fz[:, None] * c_new
    new_com[:, 2] += z_start
    pos = np.zeros((len(zif), 3))
    for u, idx in enumerate(units):
        pos[idx] = whole[idx] - coms[u] + new_com[u]   # rigid translation of each unit
    # lateral mismatch between ZIF cell and NaP cell (both lower-triangular)
    F = np.linalg.solve(cell[:2, :2], np.array([a_vec[:2], b_vec[:2]]))
    sv = np.linalg.svd(F, compute_uv=False)
    mismatch = max(abs(sv.max() - 1), abs(sv.min() - 1), abs(h / cell[2, 2] - 1))
    print(f"  ZIF layer: h = {h:.2f} A; lateral stretch {sv.min():.3f}-{sv.max():.3f}, "
          f"height x{h / cell[2, 2]:.3f} (ring geometry unchanged)")
    if mismatch > 0.02:
        print(f"  WARNING: ZIF cell differs from NaP lateral cell by {100 * mismatch:.0f}% -> "
              f"Zn-N bonds get strained/broken.\n"
              f"           For an exact ZIF layer, pre-shape the ZIF glass with "
              f"--write_zif_reshape and pass the result as --zif.")
    return sym, pos, unit_of, kinds, h, nbroken, mismatch


# ----------------------------------------------------------------------
# clash removal: rigid moves of ZIF units only
# ----------------------------------------------------------------------
def thresholds(sa, sb, dmin, dmin_h):
    hh = (sa == "H") | (sb == "H")
    zn = ((sa == "Zn") & np.isin(sb, ["N", "O"])) | ((sb == "Zn") & np.isin(sa, ["N", "O"]))
    return np.where(zn, ZN_MIN, np.where(hh, dmin_h, dmin))


def mic(d, cell, inv):
    f = d @ inv
    f -= np.round(f)
    return f @ cell


def overlap_energy(pos_u, sym_u, pos_o, sym_o, cell, inv, dmin, dmin_h):
    d = pos_u[:, None, :] - pos_o[None, :, :]
    d = mic(d.reshape(-1, 3), cell, inv).reshape(d.shape)
    r = np.sqrt((d ** 2).sum(-1))
    thr = thresholds(sym_u[:, None], sym_o[None, :], dmin, dmin_h)
    return float((np.clip(thr - r, 0, None) ** 2).sum())


def random_rotation(rng, max_angle):
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    t = rng.uniform(-max_angle, max_angle)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def remove_clashes(sym, pos, unit_ids, is_zif, cell, dmin, dmin_h, rng, sweeps=300):
    inv = np.linalg.inv(cell)
    for sweep in range(sweeps):
        i, j, d = neighbor_list("ijd", Atoms(sym.tolist(), pos, cell=cell, pbc=True),
                                max(dmin, dmin_h, ZN_MIN))
        inter = (unit_ids[i] != unit_ids[j]) & (is_zif[i] | is_zif[j])
        clash = inter & (d < thresholds(sym[i], sym[j], dmin, dmin_h))
        if not clash.any():
            return sweep, 0
        bad = {u for u in np.concatenate([unit_ids[i[clash]], unit_ids[j[clash]]])
               if is_zif[unit_ids == u][0]}
        for u in rng.permutation(sorted(bad)):
            idx = np.where(unit_ids == u)[0]
            oth = unit_ids != u
            e0 = overlap_energy(pos[idx], sym[idx], pos[oth], sym[oth], cell, inv, dmin, dmin_h)
            if e0 == 0:
                continue
            com = pos[idx].mean(0)
            best, be = pos[idx], e0
            step = 0.3 + 0.01 * sweep
            for _ in range(80):
                R = random_rotation(rng, np.radians(35)) if len(idx) > 1 else np.eye(3)
                trial = (pos[idx] - com) @ R.T + com + rng.normal(scale=step, size=3)
                e = overlap_energy(trial, sym[idx], pos[oth], sym[oth], cell, inv, dmin, dmin_h)
                if e < be:
                    best, be = trial, e
                    if e == 0:
                        break
            pos[idx] = best
    return sweeps, int(clash.sum() // 2)


# ----------------------------------------------------------------------
# main build
# ----------------------------------------------------------------------
def build(args):
    rng = np.random.default_rng(args.seed)
    zif0 = to_standard(read_structure(args.zif, ["C", "H", "N", "Zn"]))
    nap0 = to_standard(read_structure(args.nap, ["Na", "O", "P"]))
    print(f"\n{'=' * 66}\n  ZIF-62 / 40NaP layered composite (v3: NaP exact, ZIF rigid linkers)\n{'=' * 66}")
    print(f"  ZIF glass: {len(zif0)} atoms {zif0.get_chemical_formula()} rho={density_gcc(zif0):.3f}")
    print(f"  NaP glass: {len(nap0)} atoms {nap0.get_chemical_formula()} rho={density_gcc(nap0):.3f}")
    integrity_report(zif0, "ZIF input")
    rn = integrity_report(nap0, "NaP input")
    if rn.get("O-O<1.6A (O2)", 0) or set(rn.get("P_CN(O)", {})) - {4}:
        print("  WARNING: NaP input is defective (O2 or P not 4-coordinated).")

    zif = zif0.repeat((args.zif_rep[0], args.zif_rep[1], args.n_zif))
    nap = nap0.repeat((1, 1, args.n_nap))
    m_z, m_n = mass_of(zif.get_chemical_symbols()), mass_of(nap.get_chemical_symbols())
    print(f"\n  {args.zif_rep[0]}x{args.zif_rep[1]}x{args.n_zif} ZIF + {args.n_nap} NaP cells -> NaP:ZIF = {m_n / m_z:.2f}:1 by weight "
          f"(ZIF {100 * m_z / (m_z + m_n):.1f} wt%)")

    # ---- NaP slab (exact) --------------------------------------------
    ns, npos, cz, nap_broken = build_nap_slab(nap)
    a_vec, b_vec = np.array(nap.cell[0]), np.array(nap.cell[1])

    # ---- ZIF layer above it -------------------------------------------
    zs, zpos, zunit, zkinds, hz, zif_broken, mismatch = build_zif_layer(zif, a_vec, b_vec, cz + args.gap)
    Lz = cz + hz + 2 * args.gap
    cell = np.array([a_vec, b_vec, [0.0, 0.0, Lz]])
    print(f"  Box: |a|={np.linalg.norm(a_vec):.3f} |b|={np.linalg.norm(b_vec):.3f} "
          f"Lz={Lz:.3f} A  (NaP {cz:.2f} + ZIF {hz:.2f} + 2 x gap {args.gap})")

    sym = np.concatenate([zs, ns])
    pos = np.vstack([zpos, npos])
    unit_ids = np.concatenate([zunit, np.arange(len(ns)) + zunit.max() + 1])
    is_zif = np.concatenate([np.ones(len(zs), bool), np.zeros(len(ns), bool)])
    nap_before = pos[~is_zif].copy()

    print(f"\n  Removing clashes by rigid ZIF-unit moves (NaP fixed) ...")
    sweeps, left = remove_clashes(sym, pos, unit_ids, is_zif, cell,
                                  args.min_dist, args.min_dist_h, rng)
    print(f"    sweeps: {sweeps}, remaining clashes: {left}")
    assert np.array_equal(nap_before, pos[~is_zif]), "NaP atoms were moved!"

    order = np.argsort([SPECORDER.index(s) for s in sym], kind="stable")
    # "whole" copy: every linker / PO4 drawn intact (some atoms sit just outside the
    # box along z) -> use this one for visualisation (OVITO/VESTA do not draw bonds
    # across the periodic boundary)
    comp_whole = Atoms(sym.tolist(), positions=pos, cell=cell, pbc=True)[order]
    comp = comp_whole.copy()
    comp.wrap()     # wrapped copy for LAMMPS / VASP (identical under PBC)

    rho_ideal = (m_z + m_n) * 1.66054 / (zif.get_volume() + nap.get_volume())
    print(f"\n  Composite: {len(comp)} atoms {comp.get_chemical_formula()}")
    print(f"  Density: {density_gcc(comp):.3f} g/cc  (ideal two-phase: {rho_ideal:.3f})")
    print(f"  Cut bonds: {nap_broken} P-O-P bridges (-> {nap_broken} three-coordinated P), "
          f"{zif_broken} Zn-N bonds")
    rep = integrity_report(comp, "composite")
    print(f"  Closest contacts: {min_contacts(comp)}")
    n_link = sum(k in ("Im", "bIm") for k in zkinds)
    if rep.get("Im", 0) + rep.get("bIm", 0) != n_link:
        print("  WARNING: linker count changed -- inspect structure!")

    base = args.output.rsplit(".", 1)[0]
    write(args.output, comp_whole, format="extxyz")
    write(base + "_wrapped.xyz", comp, format="extxyz")
    write(base + "_POSCAR", comp, format="vasp", direct=True, sort=False)
    try:
        write(base + ".data", comp, format="lammps-data", specorder=SPECORDER,
              atom_style="atomic", masses=True)
    except TypeError:
        write(base + ".data", comp, format="lammps-data", specorder=SPECORDER, atom_style="atomic")
    with open(base + "_info.txt", "w") as f:
        f.write(f"ZIF source: {args.zif}\nNaP source: {args.nap}\n")
        f.write(f"ZIF cells: {args.n_zif}, NaP cells: {args.n_nap} (layered along z)\n")
        f.write(f"Weight ratio NaP:ZIF = {m_n / m_z:.3f}:1\n")
        f.write(f"Total atoms: {len(comp)}\nComposition: {comp.get_chemical_formula()}\n")
        f.write(f"Cell a: {a_vec.round(5).tolist()}\nCell b: {b_vec.round(5).tolist()}\n")
        f.write(f"Cell c: [0, 0, {Lz:.5f}]\n")
        f.write(f"NaP slab z: 0 - {cz:.3f} A (exact coordinates); ZIF layer z: "
                f"{cz + args.gap:.3f} - {cz + args.gap + hz:.3f} A\n")
        f.write(f"Density: {density_gcc(comp):.4f} g/cc (ideal {rho_ideal:.4f})\n")
        f.write(f"Broken at cut: {nap_broken} P-O-P, {zif_broken} Zn-N\n")
        f.write(f"Integrity: {rep}\nLAMMPS type order: {' '.join(SPECORDER)}\nSeed: {args.seed}\n")
    print(f"\n  Written: {args.output} (molecules whole, for viewing), {base}_wrapped.xyz,\n"
          f"           {base}_POSCAR, {base}.data (wrapped, for LAMMPS), {base}_info.txt")
    print("  LAMMPS: pair_coeff * * mace_matpes_r2scan_0.model-lammps.pt C H N Na O P Zn")
    print("=" * 66 + "\n")



# ----------------------------------------------------------------------
# optional pre-step: reshape the pure ZIF glass into the NaP lateral cell
# ----------------------------------------------------------------------
def write_zif_reshape(args):
    zif = to_standard(read_structure(args.zif, ["C", "H", "N", "Zn"]))
    nap = to_standard(read_structure(args.nap, ["Na", "O", "P"]))
    lx, ly = nap.cell[0, 0], nap.cell[1, 1]
    xy = nap.cell[1, 0]
    lz = zif.get_volume() / (lx * ly)
    base = args.reshape_prefix
    write(base + "_start.data", zif, format="lammps-data", specorder=["C", "H", "N", "Zn"],
          atom_style="atomic", masses=True)
    T = args.reshape_T
    n_q = int(round((T - 300.0) / args.reshape_rate * 1000))
    txt = f"""# Reshape the validated ZIF-62 glass into the 40NaP lateral cell
# target: lx={lx:.6f} ly={ly:.6f} xy={xy:.6f} (NaP a,b)  lz={lz:.6f} (ZIF volume kept)
# rings survive at {T:.0f} K (as in the pure-ZIF melt-quench); Zn-N network re-forms.
units           metal
dimension       3
boundary        p p p
atom_style      atomic
read_data       {base}_start.data

pair_style      mliap unified {args.model} 0
pair_coeff      * * C H N Zn
neigh_modify    one 3000
neighbor        0.5 bin
neigh_modify    delay 0 every 1 check yes
timestep        0.001
thermo          500
thermo_style    custom step time temp pe press lx ly lz xy xz yz density
dump            1 all custom 2000 {base}.lammpstrj id type x y z

velocity        all create 300.0 4928459 mom yes rot no dist gaussian

# 1) heat 300 -> {T:.0f} K at fixed box (10 ps)
fix             th all nvt temp 300.0 {T:.1f} 0.1
run             10000
unfix           th

# 2) deform box to NaP lateral shape at {T:.0f} K, constant volume (30 ps)
fix             th all nvt temp {T:.1f} {T:.1f} 0.1
fix             df all deform 1 x final 0.0 {lx:.6f} y final 0.0 {ly:.6f} z final 0.0 {lz:.6f} &
                xy final {xy:.6f} xz final 0.0 yz final 0.0 remap x
run             30000
unfix           df

# 3) hold at {T:.0f} K in the new box (10 ps)
run             10000
unfix           th

# 4) quench {T:.0f} -> 300 K at {args.reshape_rate} K/ps; lateral box FIXED, only z barostatted
fix             qz all npt temp {T:.1f} 300.0 0.1 z 0.0 0.0 1.0
run             {n_q}
unfix           qz

# 5) equilibrate 300 K (20 ps), lateral box fixed
fix             eq all npt temp 300.0 300.0 0.1 z 0.0 0.0 1.0
run             20000
unfix           eq

write_data      {base}_final.data
"""
    in_path = os.path.join(os.path.dirname(base), "in." + os.path.basename(base))
    with open(in_path, "w") as f:
        f.write(txt)
    print(f"  ZIF cell (std form): a={zif.cell[0].round(3)} b={zif.cell[1].round(3)} c={zif.cell[2].round(3)}")
    print(f"  Target box: lx={lx:.4f} ly={ly:.4f} xy={xy:.4f} lz={lz:.4f} xz=yz=0")
    print(f"  Wrote {base}_start.data and {in_path}")
    print(f"  After the LAMMPS run use:  --zif {base}_final.data")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zif", help="ZIF-62 glass (extxyz with cell)")
    p.add_argument("--nap", help="40NaP glass (extxyz with cell)")
    p.add_argument("--n_zif", type=int, default=1, help="ZIF cells stacked along z")
    p.add_argument("--zif_rep", nargs=2, type=int, default=[1, 1], help="lateral ZIF repeats (to match a wider NaP cell)")
    p.add_argument("--n_nap", type=int, default=1, help="NaP cells stacked along z")
    p.add_argument("--gap", type=float, default=0.5,
                   help="extra spacing per interface (A) on top of each phase's own volume")
    p.add_argument("--min_dist", type=float, default=2.0, help="min inter-unit heavy-atom distance (A)")
    p.add_argument("--min_dist_h", type=float, default=1.5, help="min inter-unit distance with H (A)")
    p.add_argument("--output", default="composite.xyz")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--write_zif_reshape", action="store_true",
                   help="write LAMMPS input that reshapes the ZIF glass into the NaP lateral cell")
    p.add_argument("--reshape_prefix", default="zif62_reshape")
    p.add_argument("--reshape_T", type=float, default=1000.0)
    p.add_argument("--reshape_rate", type=float, default=5.0, help="quench rate K/ps")
    p.add_argument("--model", default=os.path.expanduser("~/software/lammps-mliap/models/mh1_matpes_r2scan_fp32-mliap_lammps.pt"),
                   help="ML-IAP model used in the reshape input")
    p.add_argument("--check_types", default="C H N Zn",
                   help="element order of LAMMPS types when --check reads a .data file")
    p.add_argument("--check", default=None, help="only report ring / PO4 / Zn-N integrity of this file")
    a = p.parse_args()
    if a.check:
        at = read_structure(a.check, a.check_types.split())
        print(f"{a.check}: {len(at)} atoms {at.get_chemical_formula()} rho={density_gcc(at):.3f}")
        integrity_report(at, "check")
        print(f"  Closest contacts: {min_contacts(at)}")
        return
    if not (a.zif and a.nap):
        p.error("--zif and --nap are required (or use --check)")
    if a.write_zif_reshape:
        write_zif_reshape(a)
        return
    build(a)


if __name__ == "__main__":
    main()
