"""Kasarin kelas_clip.npy dengan rasio asli (213 x 424 ≈ 1 : 2).

Default 32 x 64. Opsi lebih rinci: 48 x 96.
"""

from pathlib import Path
import argparse
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def mayoritas_blok(arr, h, w):
    src_h, src_w = arr.shape
    out = np.zeros((h, w), dtype=np.int16)
    for i in range(h):
        r0 = int(i * src_h / h)
        r1 = max(r0 + 1, int((i + 1) * src_h / h))
        for j in range(w):
            c0 = int(j * src_w / w)
            c1 = max(c0 + 1, int((j + 1) * src_w / w))
            blok = arr[r0:r1, c0:c1].ravel()
            nilai, hitung = np.unique(blok, return_counts=True)
            out[i, j] = nilai[hitung.argmax()]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "data" / "processed" / "kelas_clip.npy"))
    p.add_argument("--tinggi", type=int, default=32)
    p.add_argument("--lebar", type=int, default=64)
    p.add_argument("--out-npy", default="")
    args = p.parse_args()

    src = np.load(args.input)
    if src.ndim != 2:
        raise SystemExit(f"Harus 2D, sekarang {src.shape}")

    kecil = mayoritas_blok(src.astype(np.int16), args.tinggi, args.lebar)
    out = Path(args.out_npy) if args.out_npy else (
        ROOT / "data" / "processed" / f"kelas_clip_{args.tinggi}x{args.lebar}.npy"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, kecil)

    print(f"Sumber {tuple(src.shape)} -> {kecil.shape}  n={kecil.size}")
    print(f"NPY: {out}")
    nama = {
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
    for kode, n in zip(*np.unique(kecil, return_counts=True)):
        print(f"  {int(kode):>4}  {nama.get(int(kode), '?'):16s}  {int(n)}")


if __name__ == "__main__":
    main()
