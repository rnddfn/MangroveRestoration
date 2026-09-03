# Mangrove connectivity restoration with MaskablePPO

Optimasi lokasi restorasi mangrove di Suaka Margasatwa Kuala Lupak (raster 32×64) memakai Integral Index of Connectivity (IIC) dan membandingkan empat metode:

- Random Valid Action
- Greedy ΔIIC
- PPO tanpa action masking
- MaskablePPO

Kode ini mendukung naskah skripsi. Hasil utama yang dilaporkan memakai `experiments/run_main.py`.

## Persyaratan

- Python 3.10+
- CPU cukup (latihan 50.000 step ±3–8 menit per model)

```bash
cd mangrove_rl
pip install -r requirements.txt
```

Kalau Torch hanya CPU:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Struktur

```text
mangrove_rl/
├── data/
│   ├── raw/              # GeoTIFF hasil clip
│   └── processed/        # npy siap environment
├── src/
│   ├── config.py         # budget, seed, kode kelas
│   ├── iic.py            # hitung IIC
│   ├── env.py            # Gymnasium environment
│   ├── evaluation.py     # random, greedy, evaluasi
│   └── train.py          # PPO / MaskablePPO
├── experiments/
│   ├── run_main.py       # eksperimen utama (Bab IV)
│   ├── run_hyperparams.py
│   ├── run_clipping.py
│   └── run_cost.py
├── tools/
│   ├── clip_mahakam.py
│   ├── downsample_raster.py
│   └── make_gif.py
└── outputs/
    ├── csv/
    └── gif/
```

Semua perintah dijalankan dari akar `mangrove_rl`.

## Data

File default: `data/processed/kelas_barito_64x32.npy`  
Bentuk: 32 baris × 64 kolom. Kode MapBiomas Indonesia:

| Kode | Peran |
|------|--------|
| 5, 76 | habitat (mangrove / rawa gambut) |
| 13, 21 | boleh ditanami |
| 31, 40, 33, 0 | dikunci (tambak, sawah, air, NoData) |

Setelan di `src/config.py`:

- budget 10 sel
- jarak dispersal IIC 5 sel
- 50.000 timestep, seed 0/1/2
- MlpPolicy, `ent_coef=0.01`, `gamma=1.0`

## Eksperimen utama

```bash
python experiments/run_main.py
```

Keluaran: `outputs/csv/iic_comparison.csv`

Metrik: IIC awal, IIC akhir, peningkatan IIC, std peningkatan IIC, laju aksi tidak valid, waktu latih.

Urutan hasil yang sudah diukur (jangan dibalik di naskah):

Greedy ΔIIC > MaskablePPO > Random > PPO tanpa masking.

MaskablePPO unggul atas PPO biasa (invalid 0 vs ~0,3). Greedy tetap patokan kualitas IIC, bukan target yang harus dikalahkan.

## Uji tambahan

```bash
python experiments/run_hyperparams.py --cepat
python experiments/run_clipping.py --cepat
python experiments/run_cost.py --cepat --biaya13 1 --biaya21 3 --budget 10
```

Tanpa `--cepat` = setelan penuh (lebih lama).  
`--semua-seed` = seed 0, 1, 2.

| Skrip | Isi | CSV |
|-------|-----|-----|
| `run_hyperparams.py` | timestep, entropy, lebar net, gamma | `outputs/csv/hyperparameter.csv` |
| `run_clipping.py` | clip 0.1 / 0.2 / 0.3 / dinamis | `outputs/csv/clipping.csv` |
| `run_cost.py` | biaya 13 vs 21 | `outputs/csv/uneven_cost.csv` |

Uji biaya 1:3 dan 1:8 pada raster ini **tidak** memisahkan greedy naif dan greedy ΔIIC/biaya. Jangan dilaporkan sebagai kemenangan PPO.

## GIF urutan tanam

```bash
python tools/make_gif.py --metode greedy
python tools/make_gif.py --metode mask --model path/ke/model.zip
```

File: `outputs/gif/`.

## Raster kawasan lain

```bash
python tools/clip_mahakam.py --input mapbiomas_2024.tif
python tools/downsample_raster.py --input data/raw/clip.tif --tinggi 32 --lebar 32
```

Ganti `RASTER_PATH` di `src/config.py` jika environment memakai npy baru.

## Catatan untuk naskah

- Model yang dilaporkan: MaskablePPO setelan A (`run_main.py`).
- Kontribusi yang diuji: action masking menekan aksi tidak valid dan menaikkan IIC dibanding PPO biasa.
- Greedy bukan restoptr; tidak ada jaminan optimum global.
- Jangan mengklaim PPO mengalahkan greedy pada raster 32×64 budget 10.
- Klaim waktu greedy meledak hanya untuk skenario PCI / raster besar yang belum dijalankan.
