"""Random, dua greedy, PPO, MaskablePPO pada frontier + biaya default."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEEDS, BIAYA_DEFAULT, BUDGET
from src.env import cek_raster
from src.evaluation import (
    aksi_acak,
    aksi_greedy,
    aksi_greedy_biaya,
    evaluasi_baseline,
    ringkas,
)
from src.train import latih

ENV_KW = dict(biaya=BIAYA_DEFAULT, budget=BUDGET)
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
    p.add_argument("--tanpa-rl", action="store_true", help="hanya Random dan Greedy")
    args = p.parse_args()

    cek_raster(budget=BUDGET, biaya=BIAYA_DEFAULT)
    hasil = [
        evaluasi_baseline("Random Valid Action", aksi_acak, env_kw=ENV_KW),
        evaluasi_baseline("Greedy Delta IIC", aksi_greedy, env_kw=ENV_KW),
        evaluasi_baseline("Greedy Delta IIC/Cost", aksi_greedy_biaya, env_kw=ENV_KW),
    ]
    if not args.tanpa_rl:
        hasil.append(
            ringkas("PPO (tanpa masking)", [latih("ppo", s, env_kw=ENV_KW) for s in SEEDS], False, env_kw=ENV_KW)
        )
        hasil.append(
            ringkas("MaskablePPO", [latih("mask", s, env_kw=ENV_KW) for s in SEEDS], True, env_kw=ENV_KW)
        )
    df = pd.DataFrame(hasil)[KOLOM]
    print(df.round(4))
    out = ROOT / "outputs" / "csv" / "iic_frontier.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Done. File: {out}")
