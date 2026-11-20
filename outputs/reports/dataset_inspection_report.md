# Laporan Dataset Inspection — FER-2013

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Tahap:** Tahap 1 — Dataset Inspection (*Read-Only*)  
**Lokasi Dataset:** `data/raw/train/` dan `data/raw/test/`  
**Status Modifikasi Dataset Asli:** Tidak ada perubahan atau pemindahan file pada `data/raw/`.

---

## 1. Ringkasan Eksekutif

Seluruh **35.887 file** pada `data/raw/train/` (28.709 gambar) dan `data/raw/test/` (7.178 gambar) telah diperiksa secara penuh (*100% full scan*).

- **Struktur folder sesuai dugaan** (`data/raw/train/<kategori>/` dan `data/raw/test/<kategori>/`) tanpa file lepas maupun subfolder bersarang.
- **7 kategori ekspresi wajah** ditemukan dan **100% konsisten** antara `train` dan `test`: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
- **100% gambar valid** secara teknis (0 file rusak, 0 file *truncated*, 0 file 0-byte).
- **100% seragam** pada dimensi **48 × 48 piksel**, mode warna **Grayscale 8-bit (`L`, 1 kanal)**, ekstensi **`.jpg`**, dan format aktual **JPEG**.
- **Temuan kritis untuk tahap berikutnya** (dicatat tanpa mengubah data mentah):
  1. **Ketidakseimbangan kelas (*class imbalance*) tinggi** (`happy` vs `disgust` ≈ **16,55 : 1** pada `train`).
  2. **12 citra hitam polos** (*all-zero pixel*, seluruh piksel bernilai `0`): 11 di `train`, 1 di `test`.
  3. **Duplikasi konten piksel & overlap `train`–`test`**: 1.516 grup gambar identik, termasuk 531 grup (568 file `test`) yang juga muncul di `train` serta beberapa grup dengan konflik label.
  4. **Belum ada folder `val/` terpisah**; folder `test/` berisi gabungan `PublicTest` (3.589 gambar) dan `PrivateTest` (3.589 gambar).

---

## 2. Struktur Direktori Aktual

```text
data/raw/
├── train/                  (28.709 file, ~43,09 MB; seluruhnya prefix Training_*.jpg)
│   ├── angry/              (3.995 file)
│   ├── disgust/            (436 file)
│   ├── fear/               (4.097 file)
│   ├── happy/              (7.215 file)
│   ├── neutral/            (4.965 file)
│   ├── sad/                (4.830 file)
│   └── surprise/           (3.171 file)
└── test/                   (7.178 file, ~10,80 MB; PublicTest: 3.589, PrivateTest: 3.589)
    ├── angry/              (958 file)
    ├── disgust/            (111 file)
    ├── fear/               (1.024 file)
    ├── happy/              (1.774 file)
    ├── neutral/            (1.233 file)
    ├── sad/                (1.247 file)
    └── surprise/           (831 file)
```

| Pemeriksaan Struktur | `train` | `test` | Keterangan |
|---|---:|---:|---|
| File lepas di root `data/raw/` | 0 | 0 | Hanya berisi folder `train` dan `test` |
| File lepas di dalam folder split | 0 | 0 | Bersih, tidak ada file non-gambar/metadata |
| Subfolder bersarang di dalam kategori | 0 | 0 | Struktur datar 1 tingkat per kategori |
| Pola awalan nama file | `Training_` (28.709) | `PublicTest_` (3.589), `PrivateTest_` (3.589) | Sesuai pembagian resmi FER-2013 |

---

## 3. Konsistensi Kategori `train` dan `test`

| Aspek Pemeriksaan | Hasil |
|---|---|
| Jumlah kategori di `train` | **7** (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) |
| Jumlah kategori di `test` | **7** (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) |
| Kesesuaian nama persis (*case-sensitive*) | **Konsisten (Exact Match)** |
| Kategori hanya di `train` | Tidak ada (`[]`) |
| Kategori hanya di `test` | Tidak ada (`[]`) |

---

## 4. Distribusi Jumlah Gambar per Kategori

| Kategori | Train | % Train | Test | % Test | Total | % Total | Rasio Train / Test |
|---|---:|---:|---:|---:|---:|---:|---:|
| `angry` | 3.995 | 13,92% | 958 | 13,35% | 4.953 | 13,80% | 80,66% / 19,34% |
| `disgust` | 436 | 1,52% | 111 | 1,55% | 547 | 1,52% | 79,71% / 20,29% |
| `fear` | 4.097 | 14,27% | 1.024 | 14,27% | 5.121 | 14,27% | 80,00% / 20,00% |
| `happy` | 7.215 | 25,13% | 1.774 | 24,71% | 8.989 | 25,05% | 80,26% / 19,74% |
| `neutral` | 4.965 | 17,29% | 1.233 | 17,18% | 6.198 | 17,27% | 80,11% / 19,89% |
| `sad` | 4.830 | 16,82% | 1.247 | 17,37% | 6.077 | 16,93% | 79,48% / 20,52% |
| `surprise` | 3.171 | 11,05% | 831 | 11,58% | 4.002 | 11,15% | 79,24% / 20,76% |
| **Total** | **28.709** | **100,00%** | **7.178** | **100,00%** | **35.887** | **100,00%** | **80,00% / 20,00%** |

### Analisis Distribusi & Ketimpangan Kelas
- Proporsi `train` vs `test` sangat konsisten dan terstratifikasi di sekitar **80% : 20%** untuk setiap kelas.
- Distribusi antar kelas **tidak seimbang (*imbalanced*)**:
  - Kelas terbanyak: **`happy`** (7.215 gambar di `train`, 25,13%).
  - Kelas tersedikit: **`disgust`** (436 gambar di `train`, 1,52%).
  - Rasio ketimpangan maksimum terhadap minimum (`happy : disgust`): **16,55 : 1** pada `train`, **15,98 : 1** pada `test`, dan **16,43 : 1** pada keseluruhan dataset.

---

## 5. Dimensi, Mode Warna, Format, dan Ukuran File

| Properti | `train` (28.709 gambar) | `test` (7.178 gambar) |
|---|---|---|
| Dimensi (Lebar × Tinggi) | **48 × 48** (100% seragam) | **48 × 48** (100% seragam) |
| Mode Warna / Kanal | **`L`** — Grayscale 8-bit, 1 kanal (100%) | **`L`** — Grayscale 8-bit, 1 kanal (100%) |
| Rentang Intensitas Piksel | `0` – `255` | `0` – `255` |
| Ekstensi File | `.jpg` (100%) | `.jpg` (100%) |
| Format Aktual (Decoded) | `JPEG` (100%) | `JPEG` (100%) |
| Ketidakcocokan Ekstensi vs Format | **0** | **0** |
| Ukuran File (Min / Median / Mean / Max) | 359 B / 1.578 B / 1.573,93 B / 2.392 B | 359 B / 1.580 B / 1.577,64 B / 2.483 B |
| Total Ukuran Disk | **43,09 MB** | **10,80 MB** |

---

## 6. Pemeriksaan Gambar Rusak & Anomali Kualitas Data

### 6.1 Integritas File (Gambar Rusak)
Pemeriksaan dilakukan menggunakan dua tahap untuk setiap file: validasi header (`Image.verify()`) diikuti dekode piksel penuh (`Image.load()`) dengan mode ketat (`ImageFile.LOAD_TRUNCATED_IMAGES = False`).

- **Gambar rusak / gagal dibuka / terpotong:** **0** (`train`: 0, `test`: 0).
- **File berukuran 0 byte:** **0** (`train`: 0, `test`: 0).

### 6.2 Citra Piksel Konstan (Hitam Polos / All-Zero)
Ditemukan **12 gambar** yang valid secara format JPEG namun seluruh nilai pikselnya `0` (gambar hitam polos tanpa informasi visual wajah):

- **`train` (11 gambar):**
  - `angry` (7): `Training_10131352.jpg`, `Training_28756096.jpg`, `Training_32571770.jpg`, `Training_52563817.jpg`, `Training_78540321.jpg`, `Training_96772745.jpg`, `Training_99531165.jpg`
  - `happy` (1): `Training_87607167.jpg`
  - `neutral` (1): `Training_89335926.jpg`
  - `sad` (1): `Training_48621797.jpg`
  - `surprise` (1): `Training_48403842.jpg`
- **`test` (1 gambar):**
  - `angry` (1): `PublicTest_5543497.jpg`

### 6.3 Duplikasi Konten Gambar & Overlap `train`–`test`
Analisis hash SHA-256 (byte file) dan MD5 (array piksel hasil dekode) menunjukkan hasil identik sebagai berikut:

| Metrik Duplikasi Konten | Jumlah |
|---|---:|
| Total grup gambar identik di seluruh dataset | **1.516 grup** |
| Salinan berlebih (*extra copies*) di dalam `train` | **1.236 file** |
| — Grup duplikat internal `train` (kelas sama / kelas berbeda) | 996 / **39** grup |
| Salinan berlebih (*extra copies*) di dalam `test` | **86 file** |
| — Grup duplikat internal `test` (kelas sama / kelas berbeda) | 77 / **4** grup |
| Grup gambar identik yang muncul di `train` **dan** `test` | **531 grup** (**568 file `test`**) |
| — Lintas `train`–`test` (label sama / label berbeda) | 506 / **25** grup |

Selain itu, terdapat **2 nama file identik** yang muncul pada kategori berbeda di `train`:
- `Training_8475486.jpg` (`train/angry/` dan `train/surprise/`)
- `Training_52789098.jpg` (`train/neutral/` dan `train/surprise/`)

---

## 7. Implikasi untuk Tahap Selanjutnya (Belum Dieksekusi)

1. **Penanganan Class Imbalance:** Karena rasio `happy : disgust` mencapai ~16,55 : 1, evaluasi model sebaiknya tidak hanya mengandalkan *overall accuracy*, tetapi juga **Macro F1-Score**, **Precision/Recall per kelas**, dan **Confusion Matrix**, serta mempertimbangkan *class weights*, *oversampling*, atau augmentasi terarah saat tahap preprocessing/training.
2. **Strategi Validation Split:** Dataset belum menyediakan folder `val/` terpisah. Perlu dilakukan *stratified split* dari `train` (atau memanfaatkan pemisahan `PublicTest` vs `PrivateTest` jika mengikuti protokol benchmark FER-2013 secara ketat).
3. **Kebijakan Pembersihan Data (Opsional & Terdokumentasi):** Perlu diputuskan pada tahap preprocessing/EDA apakah 12 citra hitam polos (`all-zero`) dan duplikat berlabel konflik pada `train` akan disaring di pipeline (tanpa mengubah `data/raw/`), serta mencatat keberadaan 568 file `test` yang duplikat terhadap `train` saat menginterpretasikan metrik evaluasi.
