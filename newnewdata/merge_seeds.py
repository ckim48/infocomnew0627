"""Merge a 3-seed metrics npz with its 5-seed extension into an 8-seed npz
(the five table methods; keys: acc_all / vloss_all / txmb_all, plus the
seed-mean acc curve recomputed).

Usage: python3 newnewdata/merge_seeds.py <base.npz> <ext.npz> <out.npz>
"""
import sys
import numpy as np

base, ext, out = sys.argv[1:4]
b, e = np.load(base), np.load(ext)
SCHEMES = ["Caching-assisted", "V2V-aware", "mmFedMC", "AutoFed", "Proposed"]
KEYS = ["acc_all", "vloss_all", "txmb_all"]
z = {}
for s in SCHEMES:
    for k in KEYS:
        z[f"{s}__{k}"] = np.concatenate([b[f"{s}__{k}"], e[f"{s}__{k}"]], 0)
    z[f"{s}__acc"] = z[f"{s}__acc_all"].mean(0)
np.savez_compressed(out, **z)
n = z["Proposed__acc_all"].shape[0]
print(f"merged {base} + {ext} -> {out} ({n} seeds)")
