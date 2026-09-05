"""Skenario frontier: rasio biaya 1:1/1:2/1:3 dan anggaran 10/20/30."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    BUDGET,
    DISPERSAL_DISTANCE,
    RASTER_PATH,
    RESOLUTION_CONFIG,
    SEEDS,
    SKENARIO_BIAYA,
    SKENARIO_ANGGARAN,
    TIMESTEPS,
)
from src.env import cek_raster
from src.evaluation import (
    aksi_acak,
    aksi_bridge,
    aksi_greedy,
    aksi_greedy_biaya,
    evaluasi_baseline,
    ringkas,
)
from src.train import latih

KOLOM = [
    "resolusi",
    "skenario_biaya",
    "anggaran",
    "timesteps",
    "seeds",
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


def satu_skenario(
    nama_biaya,
    biaya,
    anggaran,
    resolusi,
    env_base,
    metode=None,
    algoritma_rl=None,
    timesteps=TIMESTEPS,
    seeds=None,
):
    seeds = list(SEEDS if seeds is None else seeds)
    algoritma_rl = list(algoritma_rl or [])
    env_kw = dict(env_base, biaya=biaya, budget=anggaran)
    print(
        f"\n=== resolusi {resolusi}  biaya {nama_biaya}  "
        f"anggaran {anggaran} ==="
    )
    cek_raster(path=env_kw["path"], budget=anggaran, biaya=biaya)
    baseline = {
        "random": ("Random Valid Action", aksi_acak, None),
        "naif": ("Greedy Naif (Delta IIC)", aksi_greedy, 1),
        "cost": ("Greedy Cost (Delta IIC/Cost)", aksi_greedy_biaya, 1),
        "bridge": ("Bridge Planner", aksi_bridge, 1),
    }
    pilihan = ["random", "naif", "cost"] if metode is None else metode
    baris = []
    for nama_metode in pilihan:
        label, fungsi, n_eval = baseline[nama_metode]
        kw_eval = {} if n_eval is None else {"n": n_eval}
        baris.append(
            evaluasi_baseline(label, fungsi, env_kw=env_kw, **kw_eval)
        )
    biaya_slug = nama_biaya.replace(":", "-")
    konfigurasi_seed = "-".join(map(str, seeds))
    for jenis in algoritma_rl:
        nama_rl = "PPO (tanpa masking)" if jenis == "ppo" else "MaskablePPO"
        nama_file = "ppo" if jenis == "ppo" else "maskableppo"
        model_dan_waktu = []
        for seed in seeds:
            stem = (
                f"{nama_file}_biaya_{biaya_slug}_budget{anggaran}_"
                f"steps{timesteps}_seed{seed}"
            )
            model_path = ROOT / "outputs" / "models" / resolusi / f"{stem}.zip"
            run_name = f"{resolusi}_{stem}"
            model_dan_waktu.append(
                latih(
                    jenis,
                    seed,
                    env_kw=env_kw,
                    timesteps=timesteps,
                    model_path=model_path,
                    run_name=run_name,
                )
            )
        baris.append(
            ringkas(
                nama_rl,
                model_dan_waktu,
                jenis == "mask",
                env_kw=env_kw,
            )
        )
    for b in baris:
        b["resolusi"] = resolusi
        b["skenario_biaya"] = nama_biaya
        b["anggaran"] = anggaran
        is_rl = b["metode"] in {"PPO (tanpa masking)", "MaskablePPO"}
        b["timesteps"] = timesteps if is_rl else 0
        b["seeds"] = konfigurasi_seed if is_rl else ""
    return baris


def simpan_hasil(df, out):
    """Simpan hasil tanpa menggandakan konfigurasi eksperimen yang sama."""
    kunci = [
        "resolusi",
        "skenario_biaya",
        "anggaran",
        "timesteps",
        "seeds",
        "metode",
    ]
    if out.exists():
        lama = pd.read_csv(out)
        for kolom in KOLOM:
            if kolom not in lama:
                lama[kolom] = "" if kolom == "seeds" else 0
        lama["seeds"] = lama["seeds"].fillna("").astype(str)
        df = pd.concat([lama[KOLOM], df[KOLOM]], ignore_index=True)
    df["seeds"] = df["seeds"].fillna("").astype(str)
    df = df.drop_duplicates(subset=kunci, keep="last")
    df.to_csv(out, index=False)
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--resolusi",
        choices=RESOLUTION_CONFIG,
        help="profil raster, dispersal distance, dan budget ekuivalen",
    )
    p.add_argument("--biaya", default="1:2", choices=list(SKENARIO_BIAYA))
    p.add_argument(
        "--anggaran",
        type=int,
        help="override budget profil resolusi",
    )
    p.add_argument("--semua", action="store_true", help="semua rasio x anggaran")
    p.add_argument("--tanpa-rl", action="store_true", help="hanya random dan greedy")
    p.add_argument(
        "--metode",
        nargs="+",
        choices=("random", "naif", "cost", "bridge"),
        help="baseline yang dijalankan",
    )
    p.add_argument(
        "--rl",
        choices=("ppo", "mask", "keduanya"),
        help="algoritma RL; jika dipakai tanpa --metode, baseline tidak dijalankan",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=TIMESTEPS,
        help=f"jumlah langkah training per seed (default: {TIMESTEPS})",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"seed training, mis. --seeds 0 1 2 (default: {SEEDS})",
    )
    args = p.parse_args()

    if args.timesteps <= 0:
        p.error("--timesteps harus lebih besar dari 0")
    if args.tanpa_rl and args.rl:
        p.error("--tanpa-rl tidak dapat dipakai bersama --rl")

    if args.resolusi:
        profil = RESOLUTION_CONFIG[args.resolusi]
        resolusi = args.resolusi
        env_base = {
            "path": profil["raster_path"],
            "jarak_max": profil["dispersal_distance"],
        }
        anggaran_default = profil["budget"]
    else:
        resolusi = "default"
        env_base = {
            "path": RASTER_PATH,
            "jarak_max": DISPERSAL_DISTANCE,
        }
        anggaran_default = BUDGET

    if not env_base["path"].exists():
        p.error(f"raster tidak ditemukan: {env_base['path']}")

    if args.semua:
        pasangan = [(n, b, a) for n, b in SKENARIO_BIAYA.items() for a in SKENARIO_ANGGARAN]
    else:
        anggaran = args.anggaran if args.anggaran is not None else anggaran_default
        pasangan = [(args.biaya, SKENARIO_BIAYA[args.biaya], anggaran)]

    hasil = []
    if args.tanpa_rl:
        algoritma_rl = []
    elif args.rl == "ppo":
        algoritma_rl = ["ppo"]
    elif args.rl == "mask":
        algoritma_rl = ["mask"]
    elif args.rl == "keduanya":
        algoritma_rl = ["ppo", "mask"]
    elif args.metode is not None:
        algoritma_rl = []
    else:
        algoritma_rl = ["ppo", "mask"]

    if args.metode is not None:
        metode = args.metode
    elif args.rl is not None:
        metode = []
    else:
        metode = None

    for nama, biaya, anggaran in pasangan:
        hasil.extend(
            satu_skenario(
                nama,
                biaya,
                anggaran,
                resolusi,
                env_base,
                metode=metode,
                algoritma_rl=algoritma_rl,
                timesteps=args.timesteps,
                seeds=args.seeds,
            )
        )

    df = pd.DataFrame(hasil)[KOLOM]
    print(df.round(4))
    nama_out = (
        "iic_frontier_multiresolusi.csv"
        if args.resolusi
        else "iic_frontier_skenario.csv"
    )
    out = ROOT / "outputs" / "csv" / nama_out
    out.parent.mkdir(parents=True, exist_ok=True)
    simpan_hasil(df, out)

    print(f"Done. File: {out}")
