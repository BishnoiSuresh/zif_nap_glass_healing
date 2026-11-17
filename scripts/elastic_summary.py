#!/usr/bin/env python3
"""elastic_summary.py elastic_C_<tag>.txt [...] -- Voigt/Reuss/Hill moduli from in.elastic output.
Each data line of the file = response of the 6 stresses to one applied strain j (column j of C)."""
import sys
import numpy as np

print(f"{'state':22s} {'K_H':>6s} {'G_H':>6s} {'E_H':>6s} {'nu_H':>6s} | {'Ex':>6s} {'Ey':>6s} {'Ez':>6s} | {'C11':>6s} {'C22':>6s} {'C33':>6s} {'C44':>6s} {'C55':>6s} {'C66':>6s}  (GPa)")
for fn in sys.argv[1:]:
    rows = [list(map(float, l.split())) for l in open(fn) if l.strip() and not l.startswith("#")]
    C = np.array(rows).T            # C[i, j]
    C = 0.5 * (C + C.T)             # symmetrise (numerical noise)
    S = np.linalg.inv(C)
    KV = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[0, 2] + C[1, 2])) / 9
    GV = (C[0, 0] + C[1, 1] + C[2, 2] - (C[0, 1] + C[0, 2] + C[1, 2]) + 3 * (C[3, 3] + C[4, 4] + C[5, 5])) / 15
    KR = 1 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[0, 2] + S[1, 2]))
    GR = 15 / (4 * (S[0, 0] + S[1, 1] + S[2, 2]) - 4 * (S[0, 1] + S[0, 2] + S[1, 2]) + 3 * (S[3, 3] + S[4, 4] + S[5, 5]))
    K, G = (KV + KR) / 2, (GV + GR) / 2
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    Ed = 1 / np.diag(S)[:3]
    tag = fn.split("elastic_C_")[-1].replace(".txt", "")
    print(f"{tag:22s} {K:6.1f} {G:6.1f} {E:6.1f} {nu:6.3f} | {Ed[0]:6.1f} {Ed[1]:6.1f} {Ed[2]:6.1f} | "
          + " ".join(f"{C[k, k]:6.1f}" for k in range(6)))
    if np.any(np.linalg.eigvalsh(C) <= 0):
        print("   WARNING: C not positive definite (mechanically unstable / open crack)")
