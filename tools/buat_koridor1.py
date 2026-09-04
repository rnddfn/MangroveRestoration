"""Peta sintetis 32x64: koridor + umpan murah di tepi hutan besar.

Umpan dan celah sama-sama kelas 13 (cost 1). Greedy ΔIIC dan
Greedy ΔIIC/Cost cenderung menempel hutan besar. Kebijakan yang
benar mengisi celah menuju hutan kecil.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "processed" / "koridor_32x64.npy"


def buat():
    g = np.full((32, 64), 33, dtype=np.int16)
    # hutan besar kiri, hutan kecil kanan (jarak > ambang IIC 5)
    g[8:20, 4:12] = 5
    g[10:16, 20:26] = 5
    # celah koridor (murah)
    g[10:16, 12:20] = 13
    # umpan murah di tepi hutan besar (bukan cost 3)
    g[8:20, 2:4] = 13
    g[6:8, 4:12] = 13
    g[20:22, 4:12] = 13
    return g


if __name__ == "__main__":
    g = buat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, g)
    print(f"Disimpan {OUT}  shape={g.shape}")
    print(f"  mangrove={(g == 5).sum()}  umpan+celah13={(g == 13).sum()}  mahal21={(g == 21).sum()}")
    print("  sel lain = air (33), dikunci")
    print("Generate ulang wajib sebelum tes greedy.")
