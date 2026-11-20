# Laporan Model Training — MLP (FER2013_MLP)

**Proyek:** Klasifikasi Ekspresi Wajah Manusia pada Citra Grayscale (FER-2013)  
**Eksperimen:** `mlp` (Varian: `mlp`, `1,347,975` parameter, `15.4887 MB`)  
**Waktu Selesai (UTC):** `2026-11-20T14:31:33.549715+00:00`

---

## 1. Konfigurasi, Isolasi Data & Metrik Deployment PRD

| Parameter | Nilai |
|---|---|
| Data Training (`model.fit`) | `22,967` citra (`data/splits/train.csv`) |
| Data Validation (`validation_data`) | `5,742` citra (`data/splits/val.csv`) |
| Data Testing selama Training & Seleksi | **Tidak diakses / tidak digunakan (`0` citra)** |
| Input Shape | `[48, 48, 1]` (grayscale 1 kanal, normalisasi tunggal `[0, 1]`) |
| Total / Trainable / Non-Trainable Params | `1,347,975` / `1,346,183` / `1,792` |
| **Model Size (`model_size_mb`)** | **`15.4887 MB`** |
| **Val Inference Time (`val_inference_time_ms_per_image`)** | **`0.0438 ms/citra`** (`0.251s` total, setelah 1-batch warm-up) |
| Max Epochs / Batch Size | `50` / `64` |
| Optimizer & Initial LR | `Adam` (`lr = 0.001`) |
| Loss & Metric | `SparseCategoricalCrossentropy` / `accuracy` |
| Class Weight Mode | `smoothed` (khusus `TRAIN`) |
| Callbacks | `EarlyStopping(patience=10)`, `ReduceLROnPlateau(factor=0.5, patience=4)`, `ModelCheckpoint(val_loss)` |
| Random Seed & Device | `42` / `CPU (TensorFlow oneDNN)` |

---

## 2. Ringkasan Hasil Training Aktual

| Metrik | Best Epoch (`#50`, min `val_loss`) | Final Epoch (`#50`) |
|---|---:|---:|
| **Training Accuracy** | **39.30%** (`0.3930`) | 39.30% (`0.3930`) |
| **Validation Accuracy** | **42.67%** (`0.4267`) | 42.67% (`0.4267`) |
| **Training Loss** | **1.2688** | 1.2688 |
| **Validation Loss** | **1.5289** | 1.5289 |
| **Learning Rate** | `6.25e-05` | `6.25e-05` |

- **Total Epoch Dijalankan:** `50` / `50` (`early_stopped = False`).
- **Total Waktu Training:** `4 menit 43.3 detik` (`5.67` detik/epoch).

---

## 3. Evaluasi Detail pada Validation Set (`5.742` Citra)

- **Val Accuracy:** **42.67%** | **Val Loss:** `1.5289` | **Val Top-2 Acc:** `62.50%`
- **Val Macro Precision:** `36.91%` | **Val Macro Recall:** `38.19%`
- **Val Macro F1:** **36.21%** | **Val Weighted F1:** **40.29%**

| ID | Kelas | Val Precision | Val Recall | Val F1-Score | Support |
|---:|---|---:|---:|---:|---:|
| `0` | Angry | 31.13% | 30.04% | 30.57% | 799 |
| `1` | Disgust | 23.08% | 20.69% | 21.82% | 87 |
| `2` | Fear | 28.57% | 6.10% | 10.05% | 820 |
| `3` | Happy | 57.09% | 65.00% | 60.79% | 1,443 |
| `4` | Neutral | 38.97% | 45.92% | 42.16% | 993 |
| `5` | Sad | 33.14% | 35.20% | 34.14% | 966 |
| `6` | Surprise | 46.42% | 64.35% | 53.93% | 634 |

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

| Metrik (Validation Set) | Exp 1: Baseline CNN (`111K`, `balanced`) | `MLP` (`mlp`, `smoothed`) | Delta |
|---|---:|---:|---:|
| **Validation Accuracy** | 52.14% | **42.67%** | `-9.47 pp` |
| **Validation Loss** | 1.2583 | **1.5289** | `+0.2706` |
| **Validation Top-2 Accuracy** | 71.37% | **62.50%** | `-8.86 pp` |
| **Validation Macro F1-Score** | 45.07% | **36.21%** | `-8.86 pp` |
| **Validation Weighted F1-Score** | 50.56% | **40.29%** | `-10.27 pp` |

### Perbandingan F1-Score & Recall per Kelas (Validation Set)

| ID | Kelas | Baseline Val F1 | `MLP` Val F1 | Delta F1 | Recall (Base → Baru) |
|---:|---|---:|---:|---:|---:|
| `0` | Angry | 37.81% | **30.57%** | -7.24 pp | 38.55% → 30.04% |
| `1` | Disgust | 23.63% | **21.82%** | -1.81 pp | 66.67% → 20.69% |
| `2` | Fear | 16.45% | **10.05%** | -6.40 pp | 10.12% → 6.10% |
| `3` | Happy | 76.29% | **60.79%** | -15.49 pp | 77.13% → 65.00% |
| `4` | Neutral | 52.73% | **42.16%** | -10.56 pp | 61.33% → 45.92% |
| `5` | Sad | 39.91% | **34.14%** | -5.77 pp | 36.02% → 35.20% |
| `6` | Surprise | 68.69% | **53.93%** | -14.76 pp | 74.92% → 64.35% |


---

## 5. Artefak Tersimpan

- **Model Terbaik:** `models/fer2013_mlp.keras`
- **Checkpoint Terakhir (`--resume`):** `models/fer2013_mlp_last.keras`
- **History Training:** `outputs/reports/mlp_training_history.csv` & `outputs/reports/mlp_training_history.json`
- **Grafik:** `outputs/figures/mlp_train_val_accuracy.png`, `outputs/figures/mlp_train_val_loss.png`, `outputs/figures/mlp_val_vs_baseline.png`
