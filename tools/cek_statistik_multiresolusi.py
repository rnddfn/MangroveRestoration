"""Audit statistik raster multiresolusi sebelum eksperimen RL."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    LOCKED_CODES,
    MANGROVE_CODES,
    RESOLUTION_CONFIG,
    RESTORABLE_CODES,
)
from src.iic import (  # noqa: E402
    hitung_iic,
    jumlah_komponen_konektivitas,
    statistik_petak,
)

STRUKTUR_4_ARAH = np.array(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    dtype=bool,
)


def audit_resolusi(resolusi: str, profil: dict[str, object]) -> dict[str, object]:
    path = Path(profil["raster_path"])
    kelas = np.load(path)
    if kelas.ndim != 2:
        raise ValueError(f"{resolusi}: raster harus 2D, didapat {kelas.shape}")

    mangrove = np.isin(kelas, MANGROVE_CODES)
    restorable = np.isin(kelas, RESTORABLE_CODES)
    frontier = binary_dilation(mangrove, structure=STRUKTUR_4_ARAH) & restorable
    jumlah_patch, patch_terbesar = statistik_petak(mangrove)
    jarak = float(profil["dispersal_distance"])
    luas_lanskap = int((mangrove | restorable).sum())

    kode_dikenali = set(MANGROVE_CODES) | set(RESTORABLE_CODES) | set(LOCKED_CODES)
    kode_unik = {int(kode) for kode in np.unique(kelas)}
    kode_tidak_dikenali = sorted(kode_unik - kode_dikenali)
    mask_tidak_dikenali = np.isin(kelas, kode_tidak_dikenali)

    masalah = []
    if int(mangrove.sum()) == 0:
        masalah.append("tanpa_mangrove")
    if int(restorable.sum()) == 0:
        masalah.append("tanpa_restorable")
    if int(frontier.sum()) == 0:
        masalah.append("frontier_kosong")
    if kode_tidak_dikenali:
        masalah.append("kelas_tidak_dikenali")

    return {
        "resolusi": resolusi,
        "tinggi": int(kelas.shape[0]),
        "lebar": int(kelas.shape[1]),
        "total_sel": int(kelas.size),
        "jumlah_mangrove": int(mangrove.sum()),
        "persen_mangrove": float(mangrove.mean() * 100),
        "jumlah_restorable": int(restorable.sum()),
        "persen_restorable": float(restorable.mean() * 100),
        "jumlah_patch": jumlah_patch,
        "patch_terbesar": patch_terbesar,
        "frontier_awal": int(frontier.sum()),
        "jumlah_komponen_konektivitas": jumlah_komponen_konektivitas(
            mangrove,
            jarak,
        ),
        "dispersal_distance": jarak,
        "budget": int(profil["budget"]),
        "iic_awal": hitung_iic(mangrove, luas_lanskap, jarak),
        "kode_tidak_dikenali": ";".join(map(str, kode_tidak_dikenali)),
        "jumlah_sel_tidak_dikenali": int(mask_tidak_dikenali.sum()),
        "status": "AMAN" if not masalah else "PERLU_REVIEW",
        "catatan": ";".join(masalah),
        "raster_path": str(path),
    }


def main() -> None:
    rows = [audit_resolusi(nama, profil) for nama, profil in RESOLUTION_CONFIG.items()]
    out = ROOT / "outputs" / "csv" / "statistik_raster_multiresolusi.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    kolom = (
        "resolusi",
        "jumlah_mangrove",
        "jumlah_restorable",
        "jumlah_patch",
        "frontier_awal",
        "jumlah_komponen_konektivitas",
        "iic_awal",
        "status",
    )
    print("  ".join(f"{nama:>33}" for nama in kolom))
    for row in rows:
        nilai = [
            row["resolusi"],
            row["jumlah_mangrove"],
            row["jumlah_restorable"],
            row["jumlah_patch"],
            row["frontier_awal"],
            row["jumlah_komponen_konektivitas"],
            f"{row['iic_awal']:.10f}",
            row["status"],
        ]
        print("  ".join(f"{str(v):>33}" for v in nilai))
    print(f"\nCSV: {out}")


if __name__ == "__main__":
    main()
