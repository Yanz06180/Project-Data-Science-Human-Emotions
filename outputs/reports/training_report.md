# Laporan Model Training — FER-2013 Custom CNN

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013) Menggunakan Convolutional Neural Network  
**Tahap:** Tahap 5 — Model Training  
**Skrip Training:** `src/train.py`  
**Waktu Selesai (UTC):** `2026-11-20T00:57:42.552467+00:00`

---

## 1. Konfigurasi & Isolasi Data

| Parameter | Nilai |
|---|---|
| Data Training (`model.fit`) | `22,967` citra (`data/splits/train.csv`) |
| Data Validation (`validation_data`) | `5,742` citra (`data/splits/val.csv`) |
| Data Testing selama Training | **Tidak diakses / tidak digunakan (`0` citra)** |
| Input Shape | `[48, 48, 1]` (grayscale 1 kanal, normalisasi tunggal `[0, 1]`) |
| Augmentasi Data | Aktif hanya saat `training=True` di dalam model (`RandomFlip`, `RandomRotation ±10°`, `RandomZoom ±10%`) |
| Max Epochs | `50` |
| Batch Size | `64` |
| Optimizer & Initial LR | `Adam` (`lr = 0.001`) |
| Loss Function | `SparseCategoricalCrossentropy` |
| Metric | `accuracy` |
| Class Weight Mode | `balanced` (dihitung khusus dari distribusi `TRAIN`) |
| Callbacks | `EarlyStopping(val_loss, patience=8, restore_best_weights=True)`, `ReduceLROnPlateau(val_loss, factor=0.5, patience=3)`, `ModelCheckpoint(val_loss, save_best_only=True)` |
| Random Seed | `42` |
| Perangkat Eksekusi | `CPU (TensorFlow oneDNN)` |

---

## 2. Ringkasan Hasil Training Aktual

| Metrik | Best Epoch (`#32`, min `val_loss`) | Final Epoch (`#40`) |
|---|---:|---:|
| **Training Accuracy** | **46.58%** (`0.4658`) | 48.07% (`0.4807`) |
| **Validation Accuracy** | **52.14%** (`0.5214`) | 51.10% (`0.5110`) |
| **Training Loss** | **1.3560** | 1.3295 |
| **Validation Loss** | **1.2583** | 1.2835 |
| **Learning Rate** | `2.50e-04` | `6.25e-05` |

- **Total Epoch Dijalankan:** `40` dari maksimal `50` (`early_stopped = True`).
- **Total Waktu Training:** `25 menit 30.7 detik` (`1530.66` detik; rata-rata `38.27` detik/epoch).
- **Verifikasi `model.evaluate(val)` dengan bobot terbaik tersimpan (`models/fer2013_cnn.keras`):** `val_loss = 1.2583`, `val_accuracy = 52.14%`.

---

## 3. Bobot Kelas (`class_weight`, Khusus Training)

| `label_id` | Kelas | Bobot |
|---:|---|---:|
| `0` | `angry` | `1.0266` |
| `1` | `disgust` | `9.4011` |
| `2` | `fear` | `1.0012` |
| `3` | `happy` | `0.5684` |
| `4` | `neutral` | `0.8260` |
| `5` | `sad` | `0.8491` |
| `6` | `surprise` | `1.2933` |

---

## 4. Artefak Tersimpan

- **Model Terbaik:** `models/fer2013_cnn.keras`
- **Checkpoint Epoch Terakhir (untuk `--resume`):** `models/fer2013_cnn_last.keras`
- **History Training (CSV & JSON):** `outputs/reports/training_history.csv`, `outputs/reports/training_history.json`
- **Grafik Accuracy:** `outputs/figures/08_train_val_accuracy.png`
- **Grafik Loss:** `outputs/figures/09_train_val_loss.png`
