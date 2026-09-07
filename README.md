# Optimasi Konektivitas Restorasi Mangrove dengan MaskablePPO

Versi **V0.3** memperluas eksperimen restorasi mangrove Kuala Lupak dari satu raster menjadi tiga resolusi, menambahkan batasan frontier dan anggaran, serta membandingkan MaskablePPO dengan beberapa strategi pembanding (*baseline*).

Tujuan eksperimen adalah memilih urutan sel restorasi yang memaksimalkan peningkatan **Integral Index of Connectivity (IIC)**. IIC adalah ukuran keterhubungan habitat: semakin tinggi nilainya, semakin baik hubungan antarkelompok habitat dalam lanskap.

## Status penelitian

V0.3 masih merupakan versi eksperimen, bukan hasil final penelitian.

- Raster utama sementara: **48×96**.
- Konfigurasi utama: biaya kelas 13:21 = **1:2**, anggaran **45**, dan jarak dispersal **7,5 sel**.
- MaskablePPO 50.000 timestep telah diuji pada seed 0, 1, dan 2.
- MaskablePPO 100.000 timestep baru memiliki hasil lengkap untuk seed 0.
- Raster 48×96 dan 64×128 masih mengandung masing-masing satu dan dua sel berkode 35. Klasifikasi kode ini harus ditetapkan sebelum eksperimen final.

Klaim sementara yang didukung data adalah bahwa MaskablePPO dapat mengungguli greedy pada kasus 48×96. Hasil ini belum boleh digeneralisasi ke semua resolusi atau konfigurasi biaya.

## Metode yang dibandingkan

| Metode | Penjelasan sederhana |
|---|---|
| Random Valid Action | Memilih secara acak dari tindakan yang sah. |
| Greedy Naif | Selalu memilih kenaikan IIC terbesar pada langkah saat ini. |
| Greedy Cost | Memilih kenaikan IIC terbesar per unit biaya. |
| Bridge Planner | Mencari tindakan yang membantu membangun penghubung antarkomponen habitat. |
| PPO tanpa masking | Agen RL yang masih dapat mencoba tindakan tidak sah. |
| MaskablePPO | Agen RL yang hanya dapat memilih tindakan yang sah. |

MaskablePPO digunakan karena **dynamic action masking** menghapus pilihan yang tidak dapat dilakukan pada setiap langkah. Suatu sel hanya valid jika masih dapat direstorasi, belum dipilih, berada pada frontier habitat, dan biayanya tidak melebihi sisa anggaran.

Reward training adalah perubahan IIC setelah satu tindakan dikalikan 100. Perkalian tersebut hanya memperbesar sinyal belajar dan tidak mengubah urutan kualitas solusi. Evaluasi tetap dilaporkan menggunakan nilai IIC asli.

## Struktur proyek

```text
Mangrove/
├── data/
│   ├── raw/                         # GeoTIFF sumber
│   └── processed/
│       └── multires/                # Raster 32×64, 48×96, dan 64×128
├── experiments/
│   ├── run_frontier.py              # Runner utama V0.3
│   └── run_main.py                  # Runner utama V0.2
├── src/
│   ├── config.py                    # Resolusi, budget, biaya, dan hyperparameter
│   ├── env.py                       # Environment restorasi
│   ├── evaluation.py                # Baseline dan evaluasi model
│   ├── iic.py                       # Perhitungan konektivitas
│   ├── paths.py                     # Path proyek
│   └── train.py                     # Training PPO dan MaskablePPO
├── tools/
│   ├── buat_raster_multiresolusi.py
│   ├── cari_klip.py
│   ├── cek_statistik_multiresolusi.py
│   └── hitung_planning_headroom.py
├── outputs/                         # Hasil lokal; diabaikan Git secara default
│   ├── csv/
│   ├── models/
│   ├── tb/
│   └── docs/
└── archive/                         # Eksperimen dan artefak lama; hanya tersimpan lokal
```

Folder `archive/` tidak diperlukan untuk menjalankan V0.3 dan tercantum dalam `.gitignore`. Riwayat V0.2 yang sudah di-commit tetap dapat diakses melalui tag `v0.2`.

## Persyaratan

- Python 3.10 atau lebih baru
- CPU dapat digunakan; GPU bersifat opsional
- Ruang penyimpanan yang cukup untuk model, karena satu model MLP saat ini berukuran sekitar 124 MB

Disarankan menggunakan virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Pastikan interpreter yang aktif memiliki `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`, `rasterio`, dan dependensi lain dari `requirements.txt`.

## Menyiapkan raster multiresolusi

Raster V0.3 dibentuk dari satu GeoTIFF sumber menggunakan *majority resampling*. Setiap sel baru menerima kelas yang paling sering muncul pada blok sumbernya. Metode ini digunakan karena data tutupan lahan bersifat kategorikal.

```powershell
python tools/buat_raster_multiresolusi.py --input data/raw/KualaLupak.tif
```

Perintah tersebut menghasilkan raster berikut:

- `kelas_kualalupak_32x64.npy`
- `kelas_kualalupak_48x96.npy`
- `kelas_kualalupak_64x128.npy`

Versi GeoTIFF dan `ringkasan_multiresolusi.csv` juga dibuat untuk pemeriksaan dan reproduksibilitas.

## Audit raster sebelum RL

Jalankan audit setiap kali raster dibuat atau diubah:

```powershell
python tools/cek_statistik_multiresolusi.py
```

Audit menyimpan:

- jumlah mangrove;
- jumlah sel yang dapat direstorasi;
- jumlah patch habitat;
- jumlah frontier awal;
- jumlah komponen konektivitas;
- IIC awal;
- kode kelas yang tidak dikenali.

Ringkasan raster saat ini:

| Resolusi | Mangrove | Restorable | Patch | Frontier | Komponen | IIC awal | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| 32×64 | 293 | 339 | 29 | 101 | 3 | 0,105683 | Aman |
| 48×96 | 672 | 789 | 44 | 209 | 2 | 0,105209 | Perlu review kode 35 |
| 64×128 | 1.179 | 1.476 | 70 | 370 | 3 | 0,100955 | Perlu review kode 35 |

## Profil resolusi

Anggaran dan jarak dispersal dinaikkan bersama resolusi agar cakupan fisik eksperimen tetap lebih sebanding.

| Resolusi | Jarak dispersal | Anggaran utama |
|---|---:|---:|
| 32×64 | 5 | 20 |
| 48×96 | 7,5 | 45 |
| 64×128 | 10 | 80 |

Gunakan `--resolusi` saat menjalankan eksperimen V0.3. Menjalankan runner tanpa opsi tersebut memakai raster default lama dan tidak direkomendasikan untuk perbandingan multiresolusi.

## Menjalankan baseline

Jalankan Greedy Naif, Greedy Cost, dan Bridge pada setiap resolusi:

```powershell
python experiments/run_frontier.py --resolusi 32x64 --metode naif cost bridge
python experiments/run_frontier.py --resolusi 48x96 --metode naif cost bridge
python experiments/run_frontier.py --resolusi 64x128 --metode naif cost bridge
```

### Skenario eksploratif S2

S2 membuka kelas 13, 21, 31 (tambak), dan 40 (sawah) sebagai kandidat restorasi.
Biaya tambahannya adalah 3 untuk tambak dan 3 untuk sawah. Skenario ini hanya
untuk analisis sensitivitas, bukan rekomendasi restorasi lapangan. Hasilnya
disimpan terpisah dalam `outputs/csv/iic_frontier_s2_multiresolusi.csv`.

```powershell
python experiments/run_frontier.py --resolusi 48x96 --skenario-lahan s2 --metode naif cost bridge
python experiments/run_frontier.py --resolusi 48x96 --skenario-lahan s2 --rl mask --timesteps 100000 --seeds 0
```

Jika hasil awal S2 tertinggal dari Greedy Cost, gunakan preset yang menstabilkan
pembaruan PPO dan memberi sinyal belajar berupa kenaikan IIC per biaya. Evaluasi
akhir tetap menggunakan IIC asli.
Model diperiksa setiap 10.000 langkah dan checkpoint dengan IIC evaluasi tertinggi
yang disimpan sebagai hasil akhir.

```powershell
python experiments/run_frontier.py --resolusi 48x96 --skenario-lahan s2 --preset-rl s2-tuned --rl mask --timesteps 200000 --seeds 0
```

Tambahkan `random` jika Random Valid Action juga ingin dihitung:

```powershell
python experiments/run_frontier.py --resolusi 48x96 --metode random naif cost bridge
```

## Melatih MaskablePPO

Command utama 50.000 timestep pada raster 48×96:

```powershell
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 50000 --seeds 0
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 50000 --seeds 1
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 50000 --seeds 2
```

Eksperimen 100.000 timestep:

```powershell
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 100000 --seeds 0
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 100000 --seeds 1
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 100000 --seeds 2
```

Beberapa seed juga dapat diberikan dalam satu command, tetapi hasil evaluasinya akan diringkas menjadi satu baris gabungan:

```powershell
python experiments/run_frontier.py --resolusi 48x96 --rl mask --timesteps 50000 --seeds 0 1 2
```

Untuk membandingkan PPO tanpa masking dan MaskablePPO pada konfigurasi yang sama:

```powershell
python experiments/run_frontier.py --resolusi 48x96 --rl keduanya --timesteps 50000 --seeds 0
```

## Hasil sementara V0.3

### Baseline multiresolusi

| Resolusi | Greedy Naif | Greedy Cost | Bridge |
|---|---:|---:|---:|
| 32×64 | 0,034446 | 0,034446 | 0,034446 |
| 48×96 | 0,022661 | 0,029665 | 0,029994 |
| 64×128 | 0,018391 | 0,027837 | 0,026765 |

Angka pada tabel adalah kenaikan IIC, bukan IIC akhir. Pada 48×96, Bridge hanya 1,11% di atas Greedy Cost. Ini disebut **planning headroom**: indikasi bahwa urutan keputusan mulai berpengaruh, bukan batas optimum matematis.

### MaskablePPO pada 48×96

| Training | Seed | Kenaikan IIC |
|---|---:|---:|
| 50k | 0 | 0,031658 |
| 50k | 1 | 0,030730 |
| 50k | 2 | 0,030651 |
| 100k | 0 | 0,032444 |

Rata-rata tiga seed 50k adalah **0,031013**. Nilai ini sekitar 4,55% di atas Greedy Cost dan 3,40% di atas Bridge. Ketiga seed 50k secara individual juga berada di atas Bridge.

Hasil 100k seed 0 adalah hasil tertinggi saat ini dan sekitar 8,17% di atas Bridge. Namun, hasil satu seed belum cukup untuk menyimpulkan bahwa 100k selalu lebih baik. Seed tambahan harus diselesaikan dan dibandingkan menggunakan rata-rata serta variasi antar-seed.

Resolusi lebih tinggi tidak otomatis menghasilkan performa RL yang lebih baik. Raster 64×128 memiliki ruang aksi lebih besar, frontier lebih banyak, episode lebih panjang, dan planning headroom terhadap Greedy Cost masih negatif. Dengan jumlah timestep yang sama, agen menerima lebih sedikit pengalaman episode lengkap.

## Menghitung planning headroom

Setelah baseline lengkap tersedia pada CSV multiresolusi, jalankan:

```powershell
python tools/hitung_planning_headroom.py
```

Nilai positif berarti Bridge mengungguli greedy terbaik pada resolusi tersebut. Nilai nol atau negatif berarti belum ada bukti bahwa perencanaan jembatan memberi keuntungan dibandingkan greedy terbaik.

## Keluaran eksperimen

| Lokasi | Isi |
|---|---|
| `outputs/csv/iic_frontier_multiresolusi.csv` | Hasil baseline dan RL multiresolusi |
| `outputs/csv/statistik_raster_multiresolusi.csv` | Audit raster sebelum RL |
| `outputs/csv/planning_headroom_multiresolusi.csv` | Perbandingan Bridge dengan greedy |
| `outputs/models/<resolusi>/` | Model hasil training |
| `outputs/tb/` | Log TensorBoard untuk kurva belajar |

Runner menggunakan kombinasi resolusi, biaya, anggaran, timestep, seed, dan metode sebagai kunci pembaruan hasil. Setelah menjalankan ulang eksperimen lama, periksa CSV untuk memastikan format seed lama seperti `0.0` tidak tercatat terpisah dari format baru `0`.

Folder `outputs/` diabaikan Git untuk mencegah model besar dan log training ikut terunggah. Jika tiga CSV hasil utama ingin disimpan dalam commit versi penelitian, tambahkan secara eksplisit:

```powershell
git add -f outputs/csv/iic_frontier_multiresolusi.csv `
  outputs/csv/statistik_raster_multiresolusi.csv `
  outputs/csv/planning_headroom_multiresolusi.csv
```

Jangan tambahkan `outputs/models/` atau `outputs/tb/` ke Git biasa. Gunakan penyimpanan artefak terpisah atau Git LFS jika model perlu dipublikasikan.

## Hyperparameter utama

| Parameter | Nilai |
|---|---|
| Policy | MlpPolicy |
| Network | 256 × 256 |
| Gamma | 1,0 |
| Entropy coefficient | 0,01 |
| Parallel environment | 4 |
| Reward scale | 100 |
| Default timestep | 50.000 |
| Default seed runner | 0 |

MlpPolicy meratakan raster menjadi vektor. CnnPolicy belum menjadi konfigurasi utama dan harus diuji sebagai eksperimen arsitektur terpisah dengan preprocessing yang benar serta seed pembanding yang sama.

## Reproduksibilitas dan batasan

- Bandingkan metode pada raster, biaya, anggaran, dan jarak dispersal yang sama.
- Laporkan rata-rata dan variasi beberapa seed, bukan hanya seed terbaik.
- Jangan menganggap pengulangan evaluasi deterministik sebagai seed independen.
- Tetapkan perlakuan kode kelas 35 sebelum eksperimen final.
- Simpan kurva belajar dan urutan sel restorasi agar strategi spasial dapat diperiksa ulang.
- Penurunan jumlah patch hanya merupakan indikator penggabungan habitat; simpan juga jumlah komponen konektivitas dan perubahan IIC.
- Greedy adalah baseline kuat, tetapi bukan bukti solusi optimum global.
- Keunggulan MaskablePPO pada 48×96 belum membuktikan keunggulan pada 32×64, 64×128, atau kawasan lain.

## Versioning

- `v0.1`: implementasi awal environment dan PPO.
- `v0.2`: frontier, anggaran terbatas, biaya relatif, dan action masking pada eksperimen awal.
- `v0.3`: raster multiresolusi, audit raster, Greedy Cost, Bridge Planner, planning headroom, serta eksperimen MaskablePPO 48×96 dengan training lebih panjang.

README pada branch utama selalu menggambarkan versi terbaru. README dan kode lama tetap dapat dibaca dengan berpindah ke tag versi terkait.
