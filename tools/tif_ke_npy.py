"""Simpan GeoTIFF clip apa adanya ke .npy. Tanpa kasarin."""

from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument(
        "--out-npy",
        default=str(ROOT / "data" / "processed" / "kelas_clip.npy"),
    )
    args = p.parse_args()

    try:
        import rasterio
    except ImportError as e:
        raise SystemExit("pip install rasterio numpy") from e

    path = Path(args.input)
    with rasterio.open(path) as src:
        band = src.read(1)
        nodata = src.nodata
        print(f"Sumber: {path}")
        print(f"  bentuk={band.shape}  dtype={band.dtype}  nodata={nodata}")

    if band.ndim != 2:
        raise SystemExit("Raster harus 1 band")

    data = band.astype(np.int16)
    if nodata is not None:
        data[band == nodata] = 0

    out = Path(args.out_npy)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, data)

    print(f"NPY: {out.resolve()}  shape={data.shape}  n={data.size}")
    nama = {
        0: "NoData",
        5: "mangrove",
        13: "non-hutan lain",
        21: "pertanian lain",
        31: "tambak",
        33: "air",
        40: "sawah",
        76: "rawa gambut",
    }
    for kode, n in zip(*np.unique(data, return_counts=True)):
        print(f"  {int(kode):>4}  {nama.get(int(kode), '?'):16s}  {int(n)}")


if __name__ == "__main__":
    main()
