"""
tools/cari_klip.py

Mencari jendela klip pada raster resolusi asli (kelas_clip.npy, 213x424) yang
punya struktur "celah + umpan" nyata -- bukan karangan -- untuk jadi kandidat
"kondisi 2 nyata" sesuai Rencana B (KONTEKS_PENELITIAN.md, bagian 9B):

    9B.1  Clip jendela nyata yang jembatan kremnya jelas (bukan seluruh 213x424)
    9B.2  Warp/kasarin 32x32 atau 32x64 TANPA mengarang sel
    9B.3  --tanpa-rl dulu. Lolos sebagai "kondisi 2 nyata" hanya jika greedy
          tidak jauh di atas Random.

Definisi "celah" dan "umpan" dipakai persis seperti di KONTEKS_PENELITIAN.md
bagian 6:

    celah / koridor : jalur sel restorable di antara DUA petak mangrove
                       terpisah. Menutupnya menaikkan IIC besar di akhir
                       rantai tanam (bukan di langkah pertama) -- ini yang
                       membuat Greedy myopic gagal.
    umpan           : sel restorable menempel LANGSUNG ke patch mangrove
                       besar. DeltaIIC langkah pertama tinggi. Greedy
                       menghabiskan budget di sini duluan.

Cara kerja:
  1. Geser jendela (ukuran native, default 96x192) di atas raster resolusi
     asli, dengan stride setengah jendela.
  2. Setiap jendela di-downsample (kasarin, mayoritas per blok -- BUKAN
     mengarang sel baru) ke resolusi target (default 32x64), karena struktur
     celah/umpan bisa hilang atau muncul tergantung resolusi -- jadi yang
     dinilai adalah versi yang BENAR-BENAR akan dipakai RL, bukan versi native.
  3. Di resolusi target itu, cari pasangan patch mangrove yang terpisah tapi
     masih terhubung lewat jalur sel restorable (BFS pada grid passable =
     mangrove | restorable, bukan locked). Panjang jalur ini = perkiraan
     jumlah sel yang perlu ditanam untuk menutup celah.
  4. Hitung juga kepadatan "umpan": jumlah sel restorable yang menempel
     langsung ke patch mangrove TERBESAR (frontier langkah pertama).
  5. Skor jendela = kombinasi: ada celah dengan panjang jalur yang masuk akal
     (tidak sepele 1 langkah, tidak juga mustahil > anggaran), DAN umpan yang
     cukup banyak untuk membuat Greedy tergoda memakai budget di situ dulu.
  6. Untuk kandidat skor tertinggi, otomatis dipotong + dikasarin ke resolusi
     target, disimpan sebagai .npy terpisah, DAN langsung dicek cepat pakai
     Random vs Greedy (n_eval kecil) untuk memverifikasi kriteria 9B.3 secara
     empiris -- bukan cuma dugaan dari skor structural.

Jalankan dari root repo (folder yang berisi src/):

    python tools/cari_klip.py
    python tools/cari_klip.py --target-h 32 --target-w 64 --top-k 5
    python tools/cari_klip.py --native-h 128 --native-w 256 --stride-frac 0.5

Catatan: script ini TIDAK melatih RL sama sekali (selaras filosofi 9B.3,
cek dulu tanpa RL). RL baru dijalankan manual oleh kamu setelah kandidat
lolos kriteria "greedy tidak jauh di atas random".
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import label

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    MANGROVE_CODES,
    RESTORABLE_CODES,
    LOCKED_CODES,
    DISPERSAL_DISTANCE,
)
from src.env import RestorasiEnv  # noqa: E402
from src.evaluation import aksi_acak, aksi_greedy, evaluasi_baseline  # noqa: E402

TETANGGA = ((-1, 0), (1, 0), (0, -1), (0, 1))
STRUKTUR_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])

# Raster sumber default
RASTER_ASLI_DEFAULT = ROOT / "data" / "processed" / "kelas_clip.npy"
OUT_DIR = ROOT / "data" / "processed" / "kandidat_klip"
OUT_CSV = ROOT / "outputs" / "csv" / "kandidat_klip_skor.csv"


# Downsampling mayoritas
def kasarin_mayoritas(kelas, target_h, target_w):
    """Downsample dengan voting mayoritas per blok. Tidak mengarang sel baru
    -- tiap sel output adalah salah satu kelas yang benar-benar ada di
    bloknya (yang paling dominan)."""
    H, W = kelas.shape
    ys = np.linspace(0, H, target_h + 1).astype(int)
    xs = np.linspace(0, W, target_w + 1).astype(int)
    out = np.zeros((target_h, target_w), dtype=kelas.dtype)
    for i in range(target_h):
        for j in range(target_w):
            blok = kelas[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            if blok.size == 0:
                continue
            nilai, hitung = np.unique(blok, return_counts=True)
            out[i, j] = nilai[np.argmax(hitung)]
    return out


# Analisis struktur jendela
def _label_patch(mask):
    lbl, n = label(mask, structure=STRUKTUR_4)
    return lbl, n


def _bfs_jarak_antar_patch(passable, lbl_mangrove, id_a, id_b):
    """BFS di grid 'passable' (mangrove | restorable, BUKAN locked) dari semua
    sel patch A, cari jarak (langkah grid) minimum ke sel manapun milik patch
    B. Mengembalikan -1 kalau sama sekali tidak terhubung lewat sel passable
    (artinya dihalangi locked -- misal tambak/sawah/air -- bukan celah yang
    bisa ditutup restorasi)."""
    H, W = passable.shape
    dist = np.full((H, W), -1, dtype=np.int32)
    q = deque()
    for y, x in zip(*np.nonzero(lbl_mangrove == id_a)):
        dist[y, x] = 0
        q.append((y, x))
    target = lbl_mangrove == id_b
    while q:
        y, x = q.popleft()
        if target[y, x]:
            return int(dist[y, x])
        for dy, dx in TETANGGA:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and passable[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = dist[y, x] + 1
                q.append((ny, nx))
    return -1


def analisis_struktur(kelas, budget_acuan=20, min_frac_luas_patch=0.08):
    """Kembalikan dict statistik struktural + skor untuk satu jendela
    (di resolusi target, sudah dikasarin).

    Catatan penting: kawasan mangrove nyata sering punya BANYAK patch kecil
    (remah/noise) di samping beberapa patch besar. Kalau skor dihitung dari
    jarak TERPENDEK di antara SEMUA pasangan patch, hasilnya selalu trivial
    kecil (ada saja sepasang remah yang kebetulan berdekatan) -- padahal
    menyambungkan dua patch kecil nyaris tidak menaikkan IIC. Yang benar-benar
    dicari adalah pasangan patch BESAR yang terhubung lewat celah restorable
    dengan panjang wajar. `min_frac_luas_patch` menyaring patch remah: hanya
    patch dengan luas >= min_frac_luas_patch * luas_patch_terbesar yang
    dianggap "signifikan" dan ikut dicari pasangannya.
    """
    mangrove = np.isin(kelas, MANGROVE_CODES)
    restorable = np.isin(kelas, RESTORABLE_CODES)
    locked = np.isin(kelas, LOCKED_CODES)
    passable = mangrove | restorable

    frac_mangrove = mangrove.mean()
    frac_restorable = restorable.mean()
    frac_locked = locked.mean()

    lbl_m, n_patch = _label_patch(mangrove)
    dasar = {
        "n_patch_mangrove": int(n_patch),
        "n_patch_signifikan": 0,
        "frac_mangrove": frac_mangrove,
        "frac_restorable": frac_restorable,
        "frac_locked": frac_locked,
        "celah_terpendek": -1,
        "hadiah_luas": 0,
        "n_pasangan_celah_valid": 0,
        "umpan_frontier": 0,
        "skor": -1.0,
    }
    if n_patch < 2 or frac_mangrove < 0.02 or frac_restorable < 0.05:
        return dasar

    luas = np.bincount(lbl_m.ravel())[1:].astype(np.float64)
    luas_maks = luas.max()
    ambang = max(2.0, min_frac_luas_patch * luas_maks)
    id_signifikan = [i + 1 for i, l in enumerate(luas) if l >= ambang]
    dasar["n_patch_signifikan"] = len(id_signifikan)
    if len(id_signifikan) < 2:
        # Patch signifikan kurang
        return dasar

    urutan_patch = np.argsort(-luas) + 1  # Terbesar lebih dulu

    # Kandidat pasangan patch
    kandidat = []
    for i in range(len(id_signifikan)):
        for j in range(i + 1, len(id_signifikan)):
            ia, ib = id_signifikan[i], id_signifikan[j]
            d = _bfs_jarak_antar_patch(passable, lbl_m, ia, ib)
            if d > 0:
                hadiah = float(luas[ia - 1] * luas[ib - 1])
                kandidat.append((d, hadiah, ia, ib))

    if not kandidat:
        return dasar

    # Saring jarak realistis
    valid = [k for k in kandidat if 2 <= k[0] <= budget_acuan]
    dasar["n_pasangan_celah_valid"] = len(valid)
    if not valid:
        dasar["celah_terpendek"] = int(min(k[0] for k in kandidat))
        return dasar

    # Maksimalkan hadiah IIC
    d_terbaik, hadiah_terbaik, ia, ib = max(valid, key=lambda k: k[1])
    dasar["celah_terpendek"] = int(d_terbaik)
    dasar["hadiah_luas"] = hadiah_terbaik

    # Frontier patch terbesar
    id_terbesar = int(urutan_patch[0])
    mask_terbesar = lbl_m == id_terbesar
    umpan = np.zeros_like(restorable)
    ys, xs = np.nonzero(mask_terbesar)
    for y, x in zip(ys, xs):
        for dy, dx in TETANGGA:
            ny, nx = y + dy, x + dx
            if 0 <= ny < kelas.shape[0] and 0 <= nx < kelas.shape[1] and restorable[ny, nx]:
                umpan[ny, nx] = True
    umpan_frontier = int(umpan.sum())
    dasar["umpan_frontier"] = umpan_frontier

    # Skor gabungan
    # Skor jarak ideal
    jarak_ideal = budget_acuan / 2.0
    kedekatan_ideal = max(1.0 - abs(d_terbaik - jarak_ideal) / budget_acuan, 0.0)
    # Normalisasi hadiah
    hadiah_norm = min(hadiah_terbaik / (luas_maks ** 2 + 1e-9), 1.0)
    umpan_norm = min(umpan_frontier / 20.0, 1.0)
    dasar["skor"] = 0.4 * kedekatan_ideal + 0.4 * hadiah_norm + 0.2 * umpan_norm
    return dasar


# Pencarian jendela kandidat
def cari_kandidat(raster, native_h, native_w, target_h, target_w, stride_frac, budget_acuan, min_frac_patch):
    H, W = raster.shape
    stride_h = max(1, int(native_h * stride_frac))
    stride_w = max(1, int(native_w * stride_frac))

    baris = []
    total = 0
    for y0 in range(0, max(H - native_h, 0) + 1, stride_h):
        for x0 in range(0, max(W - native_w, 0) + 1, stride_w):
            total += 1
            jendela_native = raster[y0:y0 + native_h, x0:x0 + native_w]
            if jendela_native.shape != (native_h, native_w):
                continue
            jendela_target = kasarin_mayoritas(jendela_native, target_h, target_w)
            stat = analisis_struktur(jendela_target, budget_acuan=budget_acuan, min_frac_luas_patch=min_frac_patch)
            stat.update({
                "y0": y0, "x0": x0,
                "native_h": native_h, "native_w": native_w,
                "target_h": target_h, "target_w": target_w,
            })
            baris.append(stat)
    print(f"  {total} jendela dipindai, {len(baris)} berhasil dianalisis.")
    return pd.DataFrame(baris)


def simpan_dan_cek_kandidat(df_top, raster, top_k, n_eval_cepat, budget_acuan):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    hasil_cek = []
    for rank, row in enumerate(df_top.itertuples(), start=1):
        y0, x0 = int(row.y0), int(row.x0)
        nh, nw = int(row.native_h), int(row.native_w)
        th, tw = int(row.target_h), int(row.target_w)
        jendela_native = raster[y0:y0 + nh, x0:x0 + nw]
        jendela_target = kasarin_mayoritas(jendela_native, th, tw)

        nama = f"kandidat_{rank:02d}_y{y0}_x{x0}_{th}x{tw}.npy"
        path_out = OUT_DIR / nama
        np.save(path_out, jendela_target)

        # Evaluasi cepat baseline
        env_kw = dict(budget=budget_acuan)
        # Muat raster sementara
        hasil_random = evaluasi_baseline(
            "Random Valid Action", aksi_acak, n=n_eval_cepat,
            env_kw={**env_kw, "path": path_out},
        )
        hasil_greedy = evaluasi_baseline(
            "Greedy Delta IIC", aksi_greedy, n=n_eval_cepat,
            env_kw={**env_kw, "path": path_out},
        )
        gap = hasil_greedy["peningkatan_iic"] - hasil_random["peningkatan_iic"]
        lolos_9b3 = gap < 0.3 * max(hasil_greedy["peningkatan_iic"], 1e-9)

        hasil_cek.append({
            "rank": rank,
            "file": nama,
            "y0": y0, "x0": x0,
            "skor_struktur": row.skor,
            "celah_terpendek": row.celah_terpendek,
            "umpan_frontier": row.umpan_frontier,
            "iic_naik_random": hasil_random["peningkatan_iic"],
            "iic_naik_greedy": hasil_greedy["peningkatan_iic"],
            "gap_greedy_random": gap,
            "lolos_kriteria_9b3": lolos_9b3,
        })
        print(
            f"  [{rank}] {nama}: greedy={hasil_greedy['peningkatan_iic']:.4f} "
            f"random={hasil_random['peningkatan_iic']:.4f} gap={gap:.4f} "
            f"{'LOLOS' if lolos_9b3 else 'greedy masih dominan'}"
        )

    return pd.DataFrame(hasil_cek)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raster", type=Path, default=RASTER_ASLI_DEFAULT,
                    help="Path raster resolusi asli (kelas_clip.npy, 213x424)")
    ap.add_argument("--native-h", type=int, default=96, help="Tinggi jendela di resolusi native")
    ap.add_argument("--native-w", type=int, default=192, help="Lebar jendela di resolusi native")
    ap.add_argument("--target-h", type=int, default=32, help="Tinggi setelah dikasarin")
    ap.add_argument("--target-w", type=int, default=64, help="Lebar setelah dikasarin")
    ap.add_argument("--stride-frac", type=float, default=0.5, help="Stride sebagai fraksi ukuran jendela")
    ap.add_argument("--budget-acuan", type=int, default=20, help="Anggaran acuan untuk skor & cek cepat")
    ap.add_argument("--min-frac-patch", type=float, default=0.08,
                    help="Ambang 'signifikan': patch dgn luas < ambang*luas_terbesar diabaikan (noise)")
    ap.add_argument("--top-k", type=int, default=5, help="Berapa kandidat teratas yang dipotong & dicek")
    ap.add_argument("--n-eval-cepat", type=int, default=5, help="Jumlah episode utk cek Random/Greedy cepat")
    args = ap.parse_args()

    if not args.raster.exists():
        raise SystemExit(
            f"Raster tidak ditemukan: {args.raster}\n"
            f"Pastikan kelas_clip.npy (hasil tif_ke_npy.py) sudah ada, atau pakai --raster path/lain.npy"
        )

    print(f"Memuat raster resolusi asli: {args.raster}")
    raster = np.load(args.raster)
    print(f"  bentuk: {raster.shape}")

    print(f"\nMemindai jendela native {args.native_h}x{args.native_w} "
          f"-> target {args.target_h}x{args.target_w} (stride={args.stride_frac})...")
    df = cari_kandidat(
        raster, args.native_h, args.native_w,
        args.target_h, args.target_w, args.stride_frac, args.budget_acuan, args.min_frac_patch,
    )

    df_valid = df[df["skor"] >= 0].sort_values("skor", ascending=False)
    print(f"\n{len(df_valid)} jendela punya celah yang masuk akal (skor >= 0).")
    if df_valid.empty:
        print(
            "Tidak ada jendela dengan struktur celah+umpan pada ukuran ini. "
            "Coba perbesar --native-h/--native-w, perkecil --stride-frac, "
            "atau turunkan syarat di analisis_struktur() (mis. frac_mangrove min)."
        )
        raise SystemExit(0)

    print("\nTop kandidat berdasarkan skor struktural (sebelum verifikasi empiris):")
    print(df_valid.head(args.top_k)[
        ["y0", "x0", "n_patch_mangrove", "n_patch_signifikan", "celah_terpendek",
         "hadiah_luas", "n_pasangan_celah_valid", "umpan_frontier", "skor"]
    ].to_string(index=False))

    print(f"\nMemotong + verifikasi empiris (Random vs Greedy, n={args.n_eval_cepat}) "
          f"untuk {args.top_k} kandidat teratas...")
    df_cek = simpan_dan_cek_kandidat(
        df_valid.head(args.top_k), raster, args.top_k, args.n_eval_cepat, args.budget_acuan,
    )

    df_cek.to_csv(OUT_CSV, index=False)
    df.to_csv(OUT_CSV.with_name("kandidat_klip_semua_jendela.csv"), index=False)

    print(f"\nSelesai. File kandidat: {OUT_DIR}/")
    print(f"Ringkasan skor+cek: {OUT_CSV}")
    n_lolos = int(df_cek["lolos_kriteria_9b3"].sum())
    print(f"\n{n_lolos}/{len(df_cek)} kandidat lolos kriteria 9B.3 "
          f"(gap Greedy-Random < 30% dari ΔIIC Greedy).")
    if n_lolos > 0:
        print("Kandidat yang lolos bisa langsung dipakai untuk uji MaskablePPO penuh.")
