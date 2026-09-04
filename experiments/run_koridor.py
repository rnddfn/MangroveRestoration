"""Uji koridor sintetis: cost konflik, budget 8. Tidak menimpa hasil Kuala Lupak."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEEDS
from src.env import cek_raster
from src.evaluation import (
    aksi_acak,
    aksi_greedy,
    aksi_greedy_biaya,
    evaluasi_baseline,
    ringkas,
)
from src.train import latih

RASTER = ROOT / "data" / "processed" / "koridor_32x64.npy"
BIAYA = {13: 1, 21: 3}
ANGGARAN = 8
JARAK = 5.0

KOLOM = [
    "metode",
    "iic_awal",
    "iic_akhir",
    "peningkatan_iic",
    "std_peningkatan_iic",
    "biaya_terpakai",
    "iic_per_unit",
    "rata_sel_ditanam",
    "rata_n_petak",
    "rata_petak_max",
    "invalid_action_rate",
    "training_time_detik",
]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tanpa-rl", action="store_true")
    p.add_argument("--reward-akhir", action="store_true")
    args = p.parse_args()

    if not RASTER.exists():
        raise SystemExit("Jalankan dulu: python tools/buat_koridor.py")

    env_kw = dict(
        path=str(RASTER),
        biaya=BIAYA,
        budget=ANGGARAN,
    )
    cek_raster(path=RASTER, budget=ANGGARAN, biaya=BIAYA)

    baris = [
        evaluasi_baseline("Random Valid Action", aksi_acak, env_kw=env_kw),
        evaluasi_baseline("Greedy Delta IIC", aksi_greedy, env_kw=env_kw),
        evaluasi_baseline("Greedy Delta IIC/Cost", aksi_greedy_biaya, env_kw=env_kw),
    ]
    if not args.tanpa_rl:
        baris.append(
            ringkas(
                "PPO (tanpa masking)",
                [latih("ppo", s, env_kw=env_kw) for s in SEEDS],
                False,
                env_kw=env_kw,
            )
        )
        baris.append(
            ringkas(
                "MaskablePPO",
                [latih("mask", s, env_kw=env_kw) for s in SEEDS],
                True,
                env_kw=env_kw,
            )
        )

    df = pd.DataFrame(baris)[KOLOM]
    print(df.round(4))
    out = ROOT / "outputs" / "csv" / "iic_koridor.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Done. File: {out}")
