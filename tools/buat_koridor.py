"""Peta sintetis 32x64: dua petak + celah murah + tepi hutan mahal."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "processed" / "koridor_32x64.npy"


def buat():
    g = np.full((32, 64), 33, dtype=np.int16)
    g[8:20, 4:12] = 5
    g[10:16, 20:26] = 5
    g[10:16, 12:20] = 13
    g[8:20, 3] = 21
    g[7, 4:12] = 21
    g[20, 4:12] = 21
    return g


if __name__ == "__main__":
    g = buat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, g)
    print(f"Disimpan {OUT}  shape={g.shape}")
    print(f"  mangrove={(g == 5).sum()}  murah13={(g == 13).sum()}  mahal21={(g == 21).sum()}")
    print("  sel lain = air (33), dikunci")
