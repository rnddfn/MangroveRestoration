"""Skenario frontier: rasio biaya 1:1/1:2/1:3 dan anggaran 10/20/30."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import SEEDS, SKENARIO_BIAYA, SKENARIO_ANGGARAN
from src.env import cek_raster
from src.evaluation import (
    aksi_acak,
    aksi_greedy,
    aksi_greedy_biaya,
    evaluasi_baseline,
    ringkas,
)
from src.train import latih

KOLOM = [
    "skenario_biaya",
    "anggaran",
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


def satu_skenario(nama_biaya, biaya, anggaran, latih_rl=True):
    env_kw = dict(biaya=biaya, budget=anggaran)
    print(f"\n=== biaya {nama_biaya}  anggaran {anggaran} ===")
    cek_raster(budget=anggaran, biaya=biaya)
    baris = [
        evaluasi_baseline("Random Valid Action", aksi_acak, env_kw=env_kw),
        evaluasi_baseline("Greedy Delta IIC", aksi_greedy, env_kw=env_kw),
        evaluasi_baseline("Greedy Delta IIC/Cost", aksi_greedy_biaya, env_kw=env_kw),
    ]
    if latih_rl:
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
    for b in baris:
        b["skenario_biaya"] = nama_biaya
        b["anggaran"] = anggaran
    return baris


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--biaya", default="1:2", choices=list(SKENARIO_BIAYA))
    p.add_argument("--anggaran", type=int, default=20)
    p.add_argument("--semua", action="store_true", help="semua rasio x anggaran")
    p.add_argument("--tanpa-rl", action="store_true", help="hanya random dan greedy")
    args = p.parse_args()

    if args.semua:
        pasangan = [(n, b, a) for n, b in SKENARIO_BIAYA.items() for a in SKENARIO_ANGGARAN]
    else:
        pasangan = [(args.biaya, SKENARIO_BIAYA[args.biaya], args.anggaran)]

    hasil = []
    for nama, biaya, anggaran in pasangan:
        hasil.extend(satu_skenario(nama, biaya, anggaran, latih_rl=not args.tanpa_rl))

    df = pd.DataFrame(hasil)[KOLOM]
    print(df.round(4))
    out = ROOT / "outputs" / "csv" / "iic_frontier_skenario.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tulis_header = not out.exists()
    df.to_csv(out, mode='a', header=tulis_header, index=False)
    
    print(f"Done. File: {out}")
