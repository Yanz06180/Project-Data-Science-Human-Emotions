# Laporan Model Training — EXP2 (improved)

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Eksperimen:** `exp2` (Arsitektur: `improved`, `617,895` parameter)  
**Waktu Selesai (UTC):** `2026-11-20T09:25:59.725869+00:00`

---

## 1. Konfigurasi & Isolasi Data

| Parameter | Nilai |
|---|---|
| Data Training (`model.fit`) | `22,967` citra (`data/splits/train.csv`) |
| Data Validation (`validation_data`) | `5,742` citra (`data/splits/val.csv`) |
| Data Testing selama Training & Seleksi | **Tidak diakses / tidak digunakan (`0` citra)** |
| Input Shape | `[48, 48, 1]` (grayscale 1 kanal, normalisasi tunggal `[0, 1]`) |
| Total / Trainable / Non-Trainable Params | `617,895` / `616,231` / `1,664` |
| Max Epochs / Batch Size | `50` / `64` |
| Optimizer & Initial LR | `Adam` (`lr = 0.001`) |
| Loss & Metric | `SparseCategoricalCrossentropy` / `accuracy` |
| Class Weight Mode | `smoothed` (khusus `TRAIN`) |
| Callbacks | `EarlyStopping(patience=10)`, `ReduceLROnPlateau(factor=0.5, patience=4)`, `ModelCheckpoint(val_loss)` |
| Random Seed & Device | `42` / `CPU (TensorFlow oneDNN)` |

---

## 2. Ringkasan Hasil Training Aktual

| Metrik | Best Epoch (`#47`, min `val_loss`) | Final Epoch (`#50`) |
|---|---:|---:|
| **Training Accuracy** | **67.97%** (`0.6797`) | 67.62% (`0.6762`) |
| **Validation Accuracy** | **63.25%** (`0.6325`) | 62.50% (`0.6250`) |
| **Training Loss** | **0.7593** | 0.7550 |
| **Validation Loss** | **1.0761** | 1.0951 |
| **Learning Rate** | `6.25e-05` | `6.25e-05` |

- **Total Epoch Dijalankan:** `50` / `50` (`early_stopped = False`).
- **Total Waktu Training:** `1 jam 6 menit 9.1 detik` (`79.38` detik/epoch).

---

## 3. Evaluasi Detail pada Validation Set (`5.742` Citra)

- **Val Accuracy:** **63.25%** | **Val Loss:** `1.0761` | **Val Top-2 Acc:** `81.82%`
- **Val Macro F1:** **59.91%** | **Val Weighted F1:** **63.00%**

| ID | Kelas | Val Precision | Val Recall | Val F1-Score | Support |
|---:|---|---:|---:|---:|---:|
| `0` | Angry | 55.34% | 47.31% | 51.01% | 799 |
| `1` | Disgust | 65.52% | 43.68% | 52.41% | 87 |
| `2` | Fear | 52.33% | 38.29% | 44.23% | 820 |
| `3` | Happy | 85.98% | 85.03% | 85.51% | 1,443 |
| `4` | Neutral | 50.83% | 70.59% | 59.11% | 993 |
| `5` | Sad | 51.53% | 54.04% | 52.75% | 966 |
| `6` | Surprise | 77.66% | 71.29% | 74.34% | 634 |

### Bobot Kelas Training (`class_weight_mode = smoothed`)

| `label_id` | Kelas | Bobot |
|---:|---|---:|
| `0` | `angry` | `0.8058` |
| `1` | `disgust` | `2.4386` |
| `2` | `fear` | `0.7958` |
| `3` | `happy` | `0.5996` |
| `4` | `neutral` | `0.7228` |
| `5` | `sad` | `0.7329` |
| `6` | `surprise` | `0.9045` |

---

## 4. Perbandingan Head-to-Head pada Validation Set (5.742 Citra, Tanpa Test)

| Metrik (Validation Set) | Exp 1: Baseline (`111K`, `balanced`) | `EXP2` (`improved`, `smoothed`) | Delta |
|---|---:|---:|---:|
| **Validation Accuracy** | 52.14% | **63.25%** | `+11.11 pp` |
| **Validation Loss** | 1.2583 | **1.0761** | `-0.1823` |
| **Validation Top-2 Accuracy** | 71.37% | **81.82%** | `+10.45 pp` |
| **Validation Macro F1-Score** | 45.07% | **59.91%** | `+14.84 pp` |
| **Validation Weighted F1-Score** | 50.56% | **63.00%** | `+12.44 pp` |

### Perbandingan F1-Score & Recall per Kelas (Validation Set)

| ID | Kelas | Baseline Val F1 | `EXP2` Val F1 | Delta F1 | Recall (Base → Baru) |
|---:|---|---:|---:|---:|---:|
| `0` | Angry | 37.81% | **51.01%** | +13.20 pp | 38.55% → 47.31% |
| `1` | Disgust | 23.63% | **52.41%** | +28.79 pp | 66.67% → 43.68% |
| `2` | Fear | 16.45% | **44.23%** | +27.77 pp | 10.12% → 38.29% |
| `3` | Happy | 76.29% | **85.51%** | +9.22 pp | 77.13% → 85.03% |
| `4` | Neutral | 52.73% | **59.11%** | +6.38 pp | 61.33% → 70.59% |
| `5` | Sad | 39.91% | **52.75%** | +12.85 pp | 36.02% → 54.04% |
| `6` | Surprise | 68.69% | **74.34%** | +5.65 pp | 74.92% → 71.29% |


---

## 5. Artefak Tersimpan

- **Model Terbaik:** `models/fer2013_cnn_exp2.keras`
- **Checkpoint Terakhir (`--resume`):** `models/fer2013_cnn_exp2_last.keras`
- **History Training:** `outputs/reports/exp2_training_history.csv` & `outputs/reports/exp2_training_history.json`
- **Grafik:** `outputs/figures/13_exp2_train_val_accuracy.png`, `outputs/figures/14_exp2_train_val_loss.png`, `outputs/figures/15_val_baseline_vs_exp2.png`
