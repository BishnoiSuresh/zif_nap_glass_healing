#!/usr/bin/env python3
"""
heal_temps.py -- healing temperatures at the same REDUCED position in the Tg window as the
experiment (manuscript: Tg 40NaP ~240 C, gZ62 ~320 C, healing 240-300 C; 300 C = 75 %).

  T_heal = Tg_NaP + f * (Tg_ZIF - Tg_NaP),   f = 0.25, 0.50, 0.75 (default), rounded to 5 K

  python heal_temps.py --tg_nap 690 [--tg_zif 862] [--f 0.25 0.5 0.75]
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--tg_nap", type=float, required=True, help="MD Tg of 40NaP (K), from the nap run (tg_bayes.py)")
ap.add_argument("--tg_zif", type=float, default=862.0, help="MD Tg of gZ62 (K); 862 K = 2368-atom 1 GPa run, from the potential energy")
ap.add_argument("--f", type=float, nargs="+", default=[0.25, 0.5, 0.75])
ap.add_argument("--exp_nap", type=float, default=513.0)
ap.add_argument("--exp_zif", type=float, default=593.0)
a = ap.parse_args()
if a.tg_zif - a.tg_nap < 60:
    print(f"WARNING: MD window only {a.tg_zif - a.tg_nap:.0f} K wide")
print(f"MD window {a.tg_nap:.0f}-{a.tg_zif:.0f} K   (experiment {a.exp_nap:.0f}-{a.exp_zif:.0f} K)")
for f in a.f:
    t = a.tg_nap + f * (a.tg_zif - a.tg_nap)
    te = a.exp_nap + f * (a.exp_zif - a.exp_nap)
    print(f"  f = {f:.2f}:  THEAL = {5 * round(t / 5):.0f} K   (experimental equivalent {te - 273.15:.0f} C)")
