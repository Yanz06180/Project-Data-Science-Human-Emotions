# Laporan Data Splitting — FER-2013 (Stratified 80/20 Train/Validation)

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Tahap:** Tahap 3 — Stratified Data Splitting  
**Modul Implementasi:** `src/data_splitting.py`  
**Output Manifest & Metadata:** `data/splits/` (`train.csv`, `val.csv`, `test.csv`, `label_mapping.json`, `split_metadata.json`)  
**Status Dataset Asli (`data/raw/`):** Tidak dimodifikasi, dipindahkan, ataupun disalin (*100% read-only*).

---

## 1. Konfigurasi Splitting

| Parameter | Nilai |
|---|---|
| Sumber data splitting | `data/raw/train/` (`28.709` gambar) |
| Data yang diisolasi (tidak ikut splitting) | `data/raw/test/` (`7.178` gambar) |
| Metode splitting | Deterministic Stratified Split (Largest Remainder Allocation + `np.random.default_rng`) |
| Proporsi target | **80% Training / 20% Validation** |
| Random Seed | **`42`** |
| Format penyimpanan | CSV Manifest (path POSIX relatif) + JSON Label Mapping & Metadata pada `data/splits/` |

---

## 2. Konsistensi Label Mapping

Mapping kelas dibuat secara alfabetis deterministik dari `data/raw/train/` dan diterapkan identik pada `train.csv`, `val.csv`, serta indeks baca-saja `test.csv` (`data/splits/label_mapping.json`):

| `label_id` | `class_name` |
|---:|---|
| `0` | `angry` |
| `1` | `disgust` |
| `2` | `fear` |
| `3` | `happy` |
| `4` | `neutral` |
| `5` | `sad` |
| `6` | `surprise` |

---

## 3. Distribusi Jumlah Data per Subset & per Kelas

| `label_id` | Kategori | Raw Train Total | Training (80%) | % Train | Validation (20%) | % Val | Rasio Val / Kelas | Test Terpisah (Untouched) | % Test |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` | `angry` | 3.995 | 3.196 | 13,92% | 799 | 13,92% | 20,00% | 958 | 13,35% |
| `1` | `disgust` | 436 | 349 | 1,52% | 87 | 1,52% | 19,95% | 111 | 1,55% |
| `2` | `fear` | 4.097 | 3.277 | 14,27% | 820 | 14,28% | 20,01% | 1.024 | 14,27% |
| `3` | `happy` | 7.215 | 5.772 | 25,13% | 1.443 | 25,13% | 20,00% | 1.774 | 24,71% |
| `4` | `neutral` | 4.965 | 3.972 | 17,29% | 993 | 17,29% | 20,00% | 1.233 | 17,18% |
| `5` | `sad` | 4.830 | 3.864 | 16,82% | 966 | 16,82% | 20,00% | 1.247 | 17,37% |
| `6` | `surprise` | 3.171 | 2.537 | 11,05% | 634 | 11,04% | 19,99% | 831 | 11,58% |
| — | **Total** | **28.709** | **22.967** | **100,00%** | **5.742** | **100,00%** | **20,00%** | **7.178** | **100,00%** |

---

## 4. Hasil Verifikasi Integritas & Reproduksibilitas

| Pemeriksaan | Status | Detail Terverifikasi |
|---|---|---|
| Tidak ada file masuk ke dua subset (`train` ∩ `val`) | **LULUS** | `0` overlap filepath antara `train` (`22.967`) dan `val` (`5.742`) |
| Isolasi `data/raw/test/` | **LULUS** | `0` file `test` masuk ke `train` atau `val`; seluruh path `train`/`val` berasal dari `data/raw/train/` |
| Konservasi jumlah data mentah | **LULUS** | `22.967 + 5.742 = 28.709` (100% data `data/raw/train/` tercakup) |
| Seluruh kelas terwakili di `train` dan `val` | **LULUS** | 7 dari 7 kelas hadir pada keduanya; selisih proporsi kelas maks hanya **`0,0124` poin persen** |
| Konsistensi label mapping | **LULUS** | 100% baris `train.csv`, `val.csv`, dan `test.csv` cocok dengan `label_mapping.json` |
| Reproduksibilitas (`seed = 42`) | **LULUS** | Eksekusi ulang dengan `seed=42` menghasilkan DataFrame `train` dan `val` yang 100% identik |

### Catatan Diagnostik untuk Tahap Preprocessing
Karena dataset mentah `data/raw/train/` belum dibersihkan dari duplikat bawaan FER-2013 (`1.236` salinan identik yang ditemukan pada tahap inspeksi), pengecekan hash byte baca-saja mencatat bahwa **`381` file di `val`** (`364` hash konten unik) memiliki salinan piksel identik di `train`. Catatan ini didokumentasikan secara transparan agar kebijakan deduplikasi atau penyaringan *all-zero* dapat diputuskan secara sadar pada tahap preprocessing berikutnya tanpa mengubah dataset mentah.

---

## 5. Cara Penggunaan Modular (Tahap Preprocessing & Training)

```python
from src.data_splitting import load_label_mapping, load_train_val_splits, load_split

label_to_id, id_to_label = load_label_mapping()
train_df, val_df = load_train_val_splits()  # Tanpa menyentuh test

# Hanya dipanggil pada tahap evaluasi akhir:
test_df = load_split("test")
```
