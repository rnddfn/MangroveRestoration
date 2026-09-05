"""
Membuat raster multiresolusi Kuala Lupak dari SATU raster kelas sumber.

Default output:
  data/processed/multires/kelas_kualalupak_32x64.npy
  data/processed/multires/kelas_kualalupak_48x96.npy
  data/processed/multires/kelas_kualalupak_64x128.npy
  data/processed/multires/kelas_kualalupak_32x64.tif
  data/processed/multires/kelas_kualalupak_48x96.tif
  data/processed/multires/kelas_kualalupak_64x128.tif
  data/processed/multires/ringkasan_multiresolusi.csv

Metode resampling: majority / mode per blok sumber.
Ini cocok untuk raster kategorikal tutupan lahan; jangan gunakan bilinear/cubic.

Contoh dari root repo:
  python tools/buat_raster_multiresolusi.py \
      --input data/raw/KualaLupak.tif

Atau resolusi lain:
  python tools/buat_raster_multiresolusi.py \
      --input data/raw/KualaLupak.tif \
      --resolusi 32x64 48x96 64x128
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


NAMA_KELAS = {
    0: "NoData",
    5: "mangrove",
    13: "non-hutan lain",
    21: "pertanian lain",
    31: "tambak",
    33: "air",
    35: "kelas 35",
    40: "sawah",
    76: "rawa gambut",
}


def parse_resolusi(teks: str) -> tuple[int, int]:
    """Ubah '32x64' menjadi (32, 64) = (tinggi, lebar)."""
    try:
        h_str, w_str = teks.lower().split("x")
        h, w = int(h_str), int(w_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Resolusi '{teks}' tidak valid. Gunakan format seperti 32x64."
        ) from exc

    if h <= 0 or w <= 0:
        raise argparse.ArgumentTypeError("Tinggi dan lebar harus > 0.")
    return h, w


def mayoritas_blok(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Downsampling raster kategorikal dengan majority/mode.

    Setiap sel target mengambil kelas yang paling banyak muncul dalam area
    sumber yang dipetakan ke sel tersebut. Jika terjadi tie, np.unique
    membuat hasil deterministik dengan memilih kode numerik terkecil.

    Aturan ini sengaja sama untuk SEMUA resolusi agar perbandingan adil.
    """
    if arr.ndim != 2:
        raise ValueError(f"Raster harus 2D, didapat {arr.shape}")

    src_h, src_w = arr.shape
    out = np.zeros((target_h, target_w), dtype=np.int16)

    for i in range(target_h):
        r0 = int(i * src_h / target_h)
        r1 = max(r0 + 1, int((i + 1) * src_h / target_h))
        r1 = min(r1, src_h)

        for j in range(target_w):
            c0 = int(j * src_w / target_w)
            c1 = max(c0 + 1, int((j + 1) * src_w / target_w))
            c1 = min(c1, src_w)

            blok = arr[r0:r1, c0:c1].ravel()
            nilai, hitung = np.unique(blok, return_counts=True)
            out[i, j] = int(nilai[np.argmax(hitung)])

    return out


def statistik_kelas(arr: np.ndarray) -> dict[int, int]:
    nilai, hitung = np.unique(arr, return_counts=True)
    return {int(k): int(n) for k, n in zip(nilai, hitung)}


def cetak_statistik(label: str, arr: np.ndarray) -> None:
    print(f"\n{label}: shape={arr.shape}, n={arr.size}")
    stats = statistik_kelas(arr)
    for kode in sorted(stats):
        print(
            f"  {kode:>4}  {NAMA_KELAS.get(kode, '?'):<18} "
            f"{stats[kode]:>7} sel ({stats[kode] / arr.size * 100:6.2f}%)"
        )


def simpan_geotiff(
    out_path: Path,
    data: np.ndarray,
    *,
    crs,
    bounds,
    nodata: int | float = 0,
) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    transform = from_bounds(
        bounds.left,
        bounds.bottom,
        bounds.right,
        bounds.top,
        data.shape[1],
        data.shape[0],
    )

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "int16",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data.astype(np.int16), 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Buat beberapa resolusi raster kategorikal Kuala Lupak dari satu "
            "GeoTIFF sumber menggunakan majority resampling."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="GeoTIFF kelas sumber, mis. data/raw/KualaLupak.tif",
    )
    parser.add_argument(
        "--resolusi",
        nargs="+",
        type=parse_resolusi,
        default=[(32, 64), (48, 96), (64, 128)],
        help="Daftar tinggi x lebar. Default: 32x64 48x96 64x128",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed/multires",
        help="Folder keluaran. Default: data/processed/multires",
    )
    parser.add_argument(
        "--prefix",
        default="kelas_kualalupak",
        help="Prefix nama file keluaran.",
    )
    parser.add_argument(
        "--tanpa-tif",
        action="store_true",
        help="Hanya simpan .npy dan CSV, tanpa GeoTIFF.",
    )
    args = parser.parse_args()

    try:
        import rasterio
    except ImportError as exc:
        raise SystemExit("Install dulu: pip install rasterio numpy") from exc

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"File input tidak ditemukan: {input_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        source = src.read(1)
        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata
        source_transform = src.transform

    if source.ndim != 2:
        raise SystemExit(f"Raster harus satu band / 2D, sekarang {source.shape}")

    # Normalisasi NoData
    source_i16 = source.astype(np.int16)
    if nodata is not None:
        source_i16[source == nodata] = 0

    print(f"Sumber : {input_path}")
    print(f"Shape  : {source_i16.shape} (tinggi x lebar)")
    print(f"CRS    : {crs}")
    print(f"NoData : {nodata} -> disimpan sebagai kelas 0")
    print(f"Pixel source transform: {source_transform}")
    cetak_statistik("RASTER SUMBER", source_i16)

    ringkasan_rows: list[dict[str, object]] = []

    # Audit raster sumber
    src_stats = statistik_kelas(source_i16)
    for kode, jumlah in sorted(src_stats.items()):
        ringkasan_rows.append(
            {
                "resolusi": f"{source_i16.shape[0]}x{source_i16.shape[1]}",
                "tinggi": source_i16.shape[0],
                "lebar": source_i16.shape[1],
                "total_sel": source_i16.size,
                "kode_kelas": kode,
                "nama_kelas": NAMA_KELAS.get(kode, "?"),
                "jumlah_sel": jumlah,
                "persen": jumlah / source_i16.size * 100,
                "sumber": "asli",
            }
        )

    for h, w in args.resolusi:
        if h > source_i16.shape[0] or w > source_i16.shape[1]:
            raise SystemExit(
                f"Target {h}x{w} lebih besar dari sumber {source_i16.shape}; "
                "script ini ditujukan untuk downsampling."
            )

        kecil = mayoritas_blok(source_i16, h, w)
        stem = f"{args.prefix}_{h}x{w}"
        npy_path = out_dir / f"{stem}.npy"
        tif_path = out_dir / f"{stem}.tif"

        np.save(npy_path, kecil)
        if not args.tanpa_tif:
            simpan_geotiff(
                tif_path,
                kecil,
                crs=crs,
                bounds=bounds,
                nodata=0,
            )

        cetak_statistik(f"TARGET {h}x{w}", kecil)
        print(f"  NPY -> {npy_path}")
        if not args.tanpa_tif:
            print(f"  TIF -> {tif_path}")

        stats = statistik_kelas(kecil)
        for kode, jumlah in sorted(stats.items()):
            ringkasan_rows.append(
                {
                    "resolusi": f"{h}x{w}",
                    "tinggi": h,
                    "lebar": w,
                    "total_sel": kecil.size,
                    "kode_kelas": kode,
                    "nama_kelas": NAMA_KELAS.get(kode, "?"),
                    "jumlah_sel": jumlah,
                    "persen": jumlah / kecil.size * 100,
                    "sumber": str(input_path),
                }
            )

    csv_path = out_dir / "ringkasan_multiresolusi.csv"
    fieldnames = [
        "resolusi",
        "tinggi",
        "lebar",
        "total_sel",
        "kode_kelas",
        "nama_kelas",
        "jumlah_sel",
        "persen",
        "sumber",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ringkasan_rows)

    print(f"\nRingkasan CSV -> {csv_path}")
    print("Selesai. Gunakan file .npy sebagai input environment RL.")


if __name__ == "__main__":
    main()
