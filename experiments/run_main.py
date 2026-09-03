"""Random vs Greedy vs PPO vs MaskablePPO."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEEDS
from src.env import cek_raster
from src.evaluation import aksi_acak, aksi_greedy, evaluasi_baseline, ringkas
from src.train import latih


if __name__ == "__main__":
    cek_raster()
    hasil = [
        evaluasi_baseline("Random Valid Action", aksi_acak),
        evaluasi_baseline("Greedy Delta IIC", aksi_greedy),
        ringkas("PPO (tanpa masking)", [latih("ppo", s) for s in SEEDS], False),
        ringkas("MaskablePPO", [latih("mask", s) for s in SEEDS], True),
    ]
    kolom = [
        "metode", "iic_awal", "iic_akhir", "peningkatan_iic",
        "std_peningkatan_iic", "invalid_action_rate", "training_time_detik",
    ]
    df = pd.DataFrame(hasil)[kolom]
    print(df.round(4))
    out = ROOT / "outputs" / "csv" / "iic_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Done. File: {out}")
