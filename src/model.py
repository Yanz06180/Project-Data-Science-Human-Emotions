"""Custom Convolutional Neural Network (CNN) architecture for FER-2013 using TensorFlow & Keras.

Membangun arsitektur Custom CNN dari awal (tanpa model pretrained) untuk klasifikasi
7 kategori ekspresi wajah manusia pada citra grayscale berukuran (48, 48, 1).

Arsitektur:
1. Input Layer               : (48, 48, 1) citra grayscale ternormalisasi [0, 1].
2. Data Augmentation (ops.)  : RandomFlip, RandomRotation ringan, RandomZoom ringan
                               (hanya aktif saat training=True, otomatis nonaktif saat
                               validasi, testing, dan inferensi).
3. Convolutional Block 1     : Conv2D(32, 3x3, ReLU) -> BatchNorm -> MaxPooling2D -> Dropout.
4. Convolutional Block 2     : Conv2D(64, 3x3, ReLU) -> BatchNorm -> MaxPooling2D -> Dropout.
5. Convolutional Block 3     : Conv2D(128, 3x3, ReLU) -> BatchNorm -> MaxPooling2D -> Dropout.
6. Classification Block      : GlobalAveragePooling2D -> Dense(128, ReLU) -> Dropout.
7. Output Layer              : Dense(7, Softmax) menghasilkan distribusi probabilitas 7 kelas.

Catatan Penting Anti-Duplikasi Augmentasi:
- Jika ``use_data_augmentation=True`` (default), augmentasi ringan berjalan langsung di
  dalam graf model (hanya saat ``training=True``). Saat tahap training nanti, gunakan
  ``build_data_loaders(..., train_augment=False)`` agar augmentasi tidak terjadi dua kali.
- Jika ingin menggunakan augmentasi dari ``src/preprocessing.py`` / ``FER2013DataLoader``,
  bangun model dengan ``build_model(use_data_augmentation=False)``.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.config import (
    AugmentationConfig,
    CONV_DROPOUT_RATE,
    CONV_FILTERS,
    CONV_KERNEL_SIZE,
    DEFAULT_AUGMENTATION_CONFIG,
    DEFAULT_LEARNING_RATE,
    DENSE_DROPOUT_RATE,
    DENSE_UNITS,
    INPUT_SHAPE,
    NUM_CLASSES,
    POOL_SIZE,
    RANDOM_SEED,
)


def build_data_augmentation(
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
    seed: int = RANDOM_SEED,
) -> keras.Sequential:
    """Membangun blok data augmentation ringan berbasis Keras preprocessing layers.

    Seluruh layer augmentasi di Keras secara bawaan HANYA aktif ketika ``training=True``
    (fase pelatihan) dan dilewati secara otomatis saat ``training=False`` (validasi,
    pengujian, dan inferensi). Parameter disinkronkan dengan ``src/config.py``.
    """
    aug_layers: list[layers.Layer] = []

    # 1. Horizontal Flip ringan (cermin kiri-kanan; ekspresi wajah tetap simetris maknanya)
    if aug_config.horizontal_flip_prob > 0.0:
        aug_layers.append(
            layers.RandomFlip(
                mode="horizontal",
                seed=seed,
                name="aug_horizontal_flip",
            )
        )

    # 2. Rotasi ringan: Keras RandomRotation menggunakan fraksi dari 2*pi (360 derajat),
    #    sehingga +-10 derajat = 10 / 360 ~= 0.0278. fill_mode="nearest" mencegah sudut hitam.
    if aug_config.rotation_range_deg > 0.0:
        rotation_factor = aug_config.rotation_range_deg / 360.0
        aug_layers.append(
            layers.RandomRotation(
                factor=rotation_factor,
                fill_mode="nearest",
                interpolation="bilinear",
                seed=seed + 1,
                name="aug_light_rotation",
            )
        )

    # 3. Zoom ringan: +-10% skala spasial dengan fill_mode="nearest"
    if aug_config.zoom_range > 0.0:
        aug_layers.append(
            layers.RandomZoom(
                height_factor=(-aug_config.zoom_range, aug_config.zoom_range),
                width_factor=(-aug_config.zoom_range, aug_config.zoom_range),
                fill_mode="nearest",
                interpolation="bilinear",
                seed=seed + 2,
                name="aug_light_zoom",
            )
        )

    return keras.Sequential(aug_layers, name="data_augmentation")


def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    conv_filters: tuple[int, int, int] = CONV_FILTERS,
    kernel_size: tuple[int, int] = CONV_KERNEL_SIZE,
    pool_size: tuple[int, int] = POOL_SIZE,
    conv_dropout_rate: float = CONV_DROPOUT_RATE,
    dense_units: int = DENSE_UNITS,
    dense_dropout_rate: float = DENSE_DROPOUT_RATE,
    use_data_augmentation: bool = True,
    aug_config: AugmentationConfig = DEFAULT_AUGMENTATION_CONFIG,
    seed: int = RANDOM_SEED,
) -> keras.Model:
    """Membangun arsitektur Custom CNN dari awal untuk citra grayscale FER-2013.

    Args:
        input_shape: Dimensi citra masukan, default ``(48, 48, 1)`` (grayscale 1 kanal).
        num_classes: Jumlah kategori ekspresi wajah, default ``7``.
        conv_filters: Jumlah filter pada 3 blok konvolusi, default ``(32, 64, 128)``.
        kernel_size: Ukuran kernel konvolusi, default ``(3, 3)``.
        pool_size: Ukuran jendela MaxPooling2D, default ``(2, 2)``.
        conv_dropout_rate: Rasio dropout pada setiap blok konvolusi, default ``0.25``.
        dense_units: Jumlah neuron pada Dense layer klasifikasi, default ``128``.
        dense_dropout_rate: Rasio dropout sebelum layer output, default ``0.50``.
        use_data_augmentation: Jika ``True``, menyertakan blok augmentasi ringan yang
            hanya aktif saat ``training=True``. Set ``False`` jika augmentasi sudah
            diterapkan oleh ``FER2013DataLoader`` di tahap preprocessing.
        aug_config: Konfigurasi augmentasi ringan dari ``src.config``.
        seed: Random seed untuk reproduksibilitas inisialisasi augmentasi.

    Returns:
        Objek ``keras.Model`` yang belum dikompilasi (gunakan :func:`compile_model`).
    """
    if len(input_shape) != 3 or input_shape[-1] != 1:
        raise ValueError(
            f"Model ini khusus citra grayscale 1 kanal (H, W, 1); diterima input_shape={input_shape}"
        )
    if len(conv_filters) != 3:
        raise ValueError(f"conv_filters harus berisi tepat 3 nilai, diterima {conv_filters}")

    # =========================================================================
    # 1. INPUT LAYER
    # Menerima citra grayscale 1-kanal berukuran 48x48x1 (sudah dinormalisasi [0, 1]).
    # Tidak ada Rescaling(1./255) di dalam model agar tidak terjadi normalisasi ganda.
    # =========================================================================
    inputs = keras.Input(shape=input_shape, name="input_grayscale")
    x = inputs

    # =========================================================================
    # DATA AUGMENTATION (HANYA AKTIF SAAT TRAINING=True)
    # Memberikan variasi ringan (flip horizontal, rotasi +-10 deg, zoom +-10%)
    # selama training untuk mengurangi overfitting tanpa memengaruhi val/test.
    # =========================================================================
    if use_data_augmentation:
        data_augmentation = build_data_augmentation(aug_config=aug_config, seed=seed)
        x = data_augmentation(x)

    # =========================================================================
    # 2. CONVOLUTIONAL BLOCK 1 (32 filters)
    # Mengekstrak fitur visual tingkat rendah: tepi (edges), garis alis, kontur bibir,
    # dan transisi gradasi pencahayaan lokal pada resolusi penuh 48x48 -> 24x24.
    # =========================================================================
    x = layers.Conv2D(
        filters=conv_filters[0],
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="block1_conv",
    )(x)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name="block1_pool")(x)
    x = layers.Dropout(rate=conv_dropout_rate, name="block1_dropout")(x)

    # =========================================================================
    # 3. CONVOLUTIONAL BLOCK 2 (64 filters)
    # Menggabungkan fitur tepi menjadi pola tingkat menengah: bentuk mata, lengkungan
    # mulut, kerutan hidung, dan struktur lokal ekspresi pada resolusi 24x24 -> 12x12.
    # =========================================================================
    x = layers.Conv2D(
        filters=conv_filters[1],
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="block2_conv",
    )(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name="block2_pool")(x)
    x = layers.Dropout(rate=conv_dropout_rate, name="block2_dropout")(x)

    # =========================================================================
    # 4. CONVOLUTIONAL BLOCK 3 (128 filters)
    # Mengekstrak representasi semantik tingkat tinggi: kombinasi keseluruhan otot wajah
    # yang membedakan 7 ekspresi (angry, disgust, fear, happy, neutral, sad, surprise)
    # pada resolusi spasial ringkas 12x12 -> 6x6.
    # =========================================================================
    x = layers.Conv2D(
        filters=conv_filters[2],
        kernel_size=kernel_size,
        padding="same",
        activation="relu",
        name="block3_conv",
    )(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.MaxPooling2D(pool_size=pool_size, name="block3_pool")(x)
    x = layers.Dropout(rate=conv_dropout_rate, name="block3_dropout")(x)

    # =========================================================================
    # 5. CLASSIFICATION BLOCK
    # GlobalAveragePooling2D merangkum setiap feature map 6x6 menjadi 1 skalar rata-rata
    # (128 fitur) sehingga jauh lebih tahan overfitting dibanding Flatten(), diikuti
    # Dense(128, ReLU) untuk kombinasi fitur non-linear dan Dropout sebagai regularisasi.
    # =========================================================================
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(units=dense_units, activation="relu", name="fc_dense")(x)
    x = layers.Dropout(rate=dense_dropout_rate, name="fc_dropout")(x)

    # =========================================================================
    # 6. OUTPUT LAYER
    # Menghasilkan 7 nilai probabilitas (berjumlah 1.0) melalui aktivasi Softmax
    # yang berkorespondensi langsung dengan ID kelas 0..6 pada src/config.py.
    # =========================================================================
    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="output_probabilities",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FER2013_Custom_CNN")
    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> keras.Model:
    """Mengompilasi model CNN dengan Adam, Sparse Categorical Crossentropy, dan Accuracy.

    SparseCategoricalCrossentropy dipilih karena label kelas dari ``data/splits/*.csv``
    dan ``FER2013DataLoader`` berbentuk integer ID (0..6), bukan one-hot vector.
    """
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )
    return model


def _verify_model_behavior(model: keras.Model) -> None:
    """Verifikasi cepat (tanpa training) terhadap dimensi input/output, probabilitas, dan augmentasi."""
    import numpy as np

    # Gunakan 2 sampel nyata dari data/splits/train.csv melalui pipeline preprocessing
    from src.data_loader import build_data_loaders

    bundle = build_data_loaders(batch_size=4, train_augment=False)
    x_sample, y_sample = bundle.val_loader[0]
    assert x_sample.shape == (4, *INPUT_SHAPE), f"Bentuk input tidak sesuai: {x_sample.shape}"

    # 1. Mode inferensi / validasi (training=False): harus 100% deterministik
    preds_eval_1 = model(x_sample, training=False).numpy()
    preds_eval_2 = model(x_sample, training=False).numpy()
    assert preds_eval_1.shape == (4, NUM_CLASSES), f"Bentuk output tidak sesuai: {preds_eval_1.shape}"
    assert np.allclose(preds_eval_1, preds_eval_2, atol=1e-6), "Output mode evaluasi tidak deterministik!"
    assert np.all(preds_eval_1 >= 0.0) and np.all(preds_eval_1 <= 1.0), "Output di luar rentang [0, 1]!"
    assert np.allclose(preds_eval_1.sum(axis=1), 1.0, atol=1e-5), "Jumlah probabilitas softmax != 1!"

    # 2. Mode training (training=True): augmentasi & dropout aktif sehingga output bervariasi
    preds_train_1 = model(x_sample, training=True).numpy()
    preds_train_2 = model(x_sample, training=True).numpy()
    assert not np.allclose(preds_train_1, preds_train_2, atol=1e-6), (
        "Mode training seharusnya stokastik (augmentasi & dropout aktif)!"
    )

    # 3. Verifikasi eksplisit blok data_augmentation: aktif saat training=True, nonaktif saat training=False
    if any(layer.name == "data_augmentation" for layer in model.layers):
        aug_layer = model.get_layer("data_augmentation")
        aug_eval = aug_layer(x_sample, training=False).numpy()
        aug_train = aug_layer(x_sample, training=True).numpy()
        assert np.allclose(aug_eval, x_sample, atol=1e-6), "Augmentasi seharusnya nonaktif saat training=False!"
        assert not np.allclose(aug_train, x_sample, atol=1e-5), "Augmentasi seharusnya aktif saat training=True!"

    print("\n=== VERIFIKASI ARSITEKTUR MODEL (TANPA TRAINING) ===")
    print(f"TensorFlow version              : {tf.__version__}")
    print(f"Input batch (real val samples)  : shape={x_sample.shape}, dtype={x_sample.dtype}, range=[{x_sample.min():.4f}, {x_sample.max():.4f}]")
    print(f"Output probabilities (eval)     : shape={preds_eval_1.shape}, sum_per_sample={np.round(preds_eval_1.sum(axis=1), 6).tolist()}")
    print("Deterministik saat training=False (augmentasi & dropout OFF) : PASSED")
    print("Stokastik saat training=True (augmentasi & dropout ON)       : PASSED")
    print(f"Optimizer : {type(model.optimizer).__name__} (lr={float(model.optimizer.learning_rate.numpy())})")
    print(f"Loss      : {model.loss.name}")
    print(f"Total params         : {model.count_params():,}")
    trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable_params = int(sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights))
    print(f"Trainable params     : {trainable_params:,}")
    print(f"Non-trainable params : {non_trainable_params:,}")


def main() -> None:
    keras.utils.set_random_seed(RANDOM_SEED)
    model = build_model(use_data_augmentation=True)
    model = compile_model(model)
    model.summary(expand_nested=True)
    _verify_model_behavior(model)


if __name__ == "__main__":
    main()
