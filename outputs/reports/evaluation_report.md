# Laporan Model Evaluation — FER-2013 Custom CNN (Test Set)

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Tahap:** Tahap 6 — Model Evaluation (Evaluasi Akhir pada Data Testing)  
**Model Dievaluasi:** `models/fer2013_cnn.keras`  
**Data Testing:** `data/raw/test/` (`7,178` citra; tanpa augmentasi, tanpa training ulang, tanpa tuning)  
**Waktu Evaluasi (UTC):** `2026-11-20T01:00:26.378397+00:00`

---

## 1. Ringkasan Metrik Utama (Test Set)

| Metrik | Nilai Aktual |
|---|---:|
| **Test Accuracy** | **52.22%** (`3,748` / `7,178` benar) |
| **Test Loss (Sparse Categorical Crossentropy)** | **`1.2605`** |
| **Top-2 Accuracy** | **71.19%** |
| **Macro Avg Precision / Recall / F1-Score** | `45.46%` / `52.17%` / **`45.09%`** |
| **Weighted Avg Precision / Recall / F1-Score** | `52.10%` / `52.22%` / **`50.43%`** |
| **Rata-rata Confidence (Benar vs Salah)** | `61.74%` vs `41.31%` |

---

## 2. Performa per Kelas Ekspresi (0–6)

| ID | Kelas | Precision | Recall | F1-Score | Support | Jumlah Diprediksi |
|---:|---|---:|---:|---:|---:|---:|
| `0` | Angry | 39.49% | 43.95% | 41.60% | 958 | 1,066 |
| `1` | Disgust | 15.29% | 66.67% | 24.87% | 111 | 484 |
| `2` | Fear | 32.23% | 7.62% | 12.32% | 1,024 | 242 |
| `3` | Happy | 76.51% | 77.28% | 76.89% | 1,774 | 1,792 |
| `4` | Neutral | 44.99% | 61.56% | 51.99% | 1,233 | 1,687 |
| `5` | Sad | 45.78% | 35.20% | 39.80% | 1,247 | 959 |
| `6` | Surprise | 63.92% | 72.92% | 68.13% | 831 | 948 |

---

## 3. Classification Report Lengkap

```text
Classification Report — FER-2013 Custom CNN (Test Set)
====================================================================
Class              Precision      Recall    F1-Score     Support
--------------------------------------------------------------------
0 = Angry             0.3949      0.4395      0.4160         958
1 = Disgust           0.1529      0.6667      0.2487         111
2 = Fear              0.3223      0.0762      0.1232        1024
3 = Happy             0.7651      0.7728      0.7689        1774
4 = Neutral           0.4499      0.6156      0.5199        1233
5 = Sad               0.4578      0.3520      0.3980        1247
6 = Surprise          0.6392      0.7292      0.6813         831
--------------------------------------------------------------------
accuracy                                      0.5222        7178
macro avg             0.4546      0.5217      0.4509        7178
weighted avg          0.5210      0.5222      0.5043        7178
====================================================================
```

---

## 4. Pasangan Ekspresi Paling Sering Tertukar

| Peringkat | Pola Kekeliruan (True → Predicted) | Jumlah | Proporsi dari Kelas Aktual |
|---:|---|---:|---:|
| 1 | **Sad → Neutral** | `328` | 26.30% dari kelas `Sad` |
| 2 | **Fear → Neutral** | `225` | 21.97% dari kelas `Fear` |
| 3 | **Sad → Angry** | `200` | 16.04% dari kelas `Sad` |
| 4 | **Fear → Sad** | `194` | 18.95% dari kelas `Fear` |
| 5 | **Angry → Neutral** | `190` | 19.83% dari kelas `Angry` |

---

## 5. Analisis Hasil Evaluasi

- **Performa Keseluruhan:** Model Custom CNN (`110.983` parameter) mencapai **Test Accuracy 52.22%**, **Macro F1-Score 45.09%**, dan **Weighted F1-Score 50.43%** (dengan Top-2 Accuracy **71.19%**), jauh melampaui baseline tebakan kelas mayoritas (`24,71%` accuracy / `~5,66%` Macro F1 pada test).
- **Kelas dengan Performa Terbaik:** **`Happy`** meraih F1-Score tertinggi sebesar **76.89%** (Precision `76.51%`, Recall `77.28%`, support `1,774`); Recall tertinggi diraih oleh **`Happy`** (`77.28%`).
- **Kelas dengan Performa Terendah:** **`Fear`** mencatat F1-Score terendah sebesar **12.32%** (Precision `32.23%`, Recall `7.62%`, support `1,024`); Recall terendah berada pada **`Fear`** (`7.62%`).
- **Ekspresi Paling Sering Tertukar:** Kekeliruan terbanyak secara absolut terjadi pada **Sad → Neutral** (`328` sampel; `26.30%` dari `Sad`), diikuti **Fear → Neutral** (`225` sampel) dan **Sad → Angry** (`200` sampel).
- **Kemungkinan Penyebab Kesalahan Klasifikasi:**
  1. **Kemiripan fitur mikro wajah antar emosi negatif/berdekatan** pada resolusi rendah `48×48` grayscale (misalnya ketegangan alis/mulut antara `Fear`, `Sad`, `Angry`, `Neutral`, serta mata/mulut terbuka antara `Fear` dan `Surprise`).
  2. **Ketidakseimbangan kelas ekstrem & efek `class_weight`:** Kelas minoritas `Disgust` (hanya `349` citra latih / `1,52%`) diberi bobot tinggi (`9,40×`) saat training sehingga model lebih sensitif memprediksi `Disgust` (menaikkan recall namun dapat menekan precision dan menggeser sebagian prediksi `Angry`/`Fear`/`Sad`).
  3. **Noise inheren dataset FER-2013** (yang ditemukan saat inspeksi & EDA): ambiguitas pelabelan manusia, duplikat lintas kelas, oklusi wajah oleh tangan, citra non-foto/watermark, serta resolusi kecil `48×48`.
  4. **Kapasitas arsitektur ringkas (`111K` parameter):** 3 blok konvolusi dengan satu `Conv2D` per blok membatasi kemampuan mengekstrak nuansa ekspresi wajah yang sangat halus dibanding blok konvolusi ganda/lebih dalam.

---

## 6. Artefak Evaluasi Tersimpan

- Confusion Matrix Heatmap: `outputs/figures/10_test_confusion_matrix.png`
- Grafik Metrik per Kelas: `outputs/figures/11_test_per_class_metrics.png`
- Analisis Kesalahan Klasifikasi: `outputs/figures/12_test_misclassification_analysis.png`
- Classification Report (TXT): `outputs/reports/classification_report.txt`
- Ringkasan Metrik (JSON): `outputs/reports/evaluation_metrics.json`
