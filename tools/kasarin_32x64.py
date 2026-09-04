"""
Kasarin raster kelas (hasil clip QGIS) menjadi 32 baris x 64 kolom
dengan aturan mayoritas (mode), lalu simpan GeoTIFF + .npy.

Contoh:
  python kasarin_32x64.py --input mahakam_2024_clip.tif
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

TARGET_H = 48
TARGET_W = 96


def mayoritas_blok(arr: np.ndarray, h: int, w: int) -> np.ndarray:
    """Setiap sel keluaran = kode yang paling banyak di blok sumbernya."""
    src_h, src_w = arr.shape
    out = np.zeros((h, w), dtype=arr.dtype)
    for i in range(h):
        r0 = int(i * src_h / h)
        r1 = int((i + 1) * src_h / h)
        for j in range(w):
            c0 = int(j * src_w / w)
            c1 = int((j + 1) * src_w / w)
            blok = arr[r0:r1, c0:c1].ravel()
            if blok.size == 0:
                continue
            nilai, hitung = np.unique(blok, return_counts=True)
            out[i, j] = nilai[hitung.argmax()]
    return out


def simpan_tif(path, data, crs, transform, nodata):
    import rasterio

    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": data.dtype,
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="GeoTIFF hasil clip")
    p.add_argument("--out-tif", default="KualaLupak.tif")
    p.add_argument("--out-npy", default="KualaLupak.npy")
    p.add_argument("--tinggi", type=int, default=TARGET_H)
    p.add_argument("--lebar", type=int, default=TARGET_W)
    args = p.parse_args()

    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError as e:
        raise SystemExit("Install dulu: pip install rasterio numpy") from e

    src_path = Path(args.input)
    with rasterio.open(src_path) as src:
        band = src.read(1)
        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata
        print(f"Sumber: {src_path}")
        print(f"  bentuk={band.shape}  CRS={crs}  nodata={nodata}")

    if band.ndim != 2:
        raise SystemExit(f"Raster harus 1 band. Bentuk sekarang: {band.shape}")

    kecil = mayoritas_blok(band, args.tinggi, args.lebar)
    transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top,
                            kecil.shape[1], kecil.shape[0])

    tif_path = Path(args.out_tif)
    npy_path = Path(args.out_npy)
    simpan_tif(tif_path, kecil, crs, transform, nodata if nodata is not None else 0)
    np.save(npy_path, kecil)

    print(f"Keluaran: {kecil.shape} (baris x kolom)")
    print(f"  TIF: {tif_path.resolve()}")
    print(f"  NPY: {npy_path.resolve()}")
    print("  rincian kelas:")
    for kode, n in zip(*np.unique(kecil, return_counts=True)):
        print(f"    {int(kode):>4}  {int(n)} sel")


if __name__ == "__main__":
    main()
