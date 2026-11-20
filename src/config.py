"""Central configuration for the FER-2013 Facial Expression Classification project.

Provides a single source of truth for directory paths, image dimensions,
pixel normalization bounds, canonical label mappings, splitting parameters,
and training-only light data augmentation settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (original raw dataset is strictly read-only)
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
RAW_TRAIN_DIR: Path = RAW_DATA_DIR / "train"
RAW_TEST_DIR: Path = RAW_DATA_DIR / "test"
SPLITS_DIR: Path = DATA_DIR / "splits"
PROCESSED_DIR: Path = DATA_DIR / "processed"
TRAIN_VAL_CACHE_PATH: Path = PROCESSED_DIR / "fer2013_train_val_uint8_cache.npz"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"
MODELS_DIR: Path = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Image & tensor specifications
# ---------------------------------------------------------------------------
IMAGE_HEIGHT: int = 48
IMAGE_WIDTH: int = 48
IMAGE_SIZE: tuple[int, int] = (IMAGE_WIDTH, IMAGE_HEIGHT)  # (W, H) for PIL
IMAGE_CHANNELS: int = 1
INPUT_SHAPE: tuple[int, int, int] = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)  # (48, 48, 1)

RAW_PIXEL_MIN: float = 0.0
RAW_PIXEL_MAX: float = 255.0
NORMALIZED_PIXEL_MIN: float = 0.0
NORMALIZED_PIXEL_MAX: float = 1.0

VALID_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

# ---------------------------------------------------------------------------
# Canonical label mapping (consistent across inspection, splitting, preprocessing, training)
# ---------------------------------------------------------------------------
LABEL_TO_ID: dict[str, int] = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6,
}
ID_TO_LABEL: dict[int, str] = {idx: name for name, idx in LABEL_TO_ID.items()}
CLASS_NAMES: tuple[str, ...] = tuple(LABEL_TO_ID.keys())
NUM_CLASSES: int = len(LABEL_TO_ID)

# ---------------------------------------------------------------------------
# Reproducibility & splitting configuration
# ---------------------------------------------------------------------------
RANDOM_SEED: int = 42
TRAIN_RATIO: float = 0.80
VAL_RATIO: float = 0.20

# ---------------------------------------------------------------------------
# Data loader & light augmentation configuration (TRAINING ONLY)
# ---------------------------------------------------------------------------
DEFAULT_BATCH_SIZE: int = 64

# ---------------------------------------------------------------------------
# Custom CNN architecture & compilation defaults (used in src/model.py)
# ---------------------------------------------------------------------------
CONV_FILTERS: tuple[int, int, int] = (32, 64, 128)
CONV_KERNEL_SIZE: tuple[int, int] = (3, 3)
POOL_SIZE: tuple[int, int] = (2, 2)
CONV_DROPOUT_RATE: float = 0.25
DENSE_UNITS: int = 128
DENSE_DROPOUT_RATE: float = 0.50
DEFAULT_LEARNING_RATE: float = 1e-3

# Improved Custom CNN (Experiment 2+) defaults
IMPROVED_CONV_FILTERS: tuple[int, int, int, int] = (32, 64, 128, 256)
IMPROVED_BLOCK_DROPOUTS: tuple[float, float, float] = (0.20, 0.25, 0.30)
IMPROVED_DENSE_DROPOUT: float = 0.45
IMPROVED_L2_WEIGHT_DECAY: float = 1e-4
IMPROVED_TRANSLATION_RANGE: float = 0.05  # +-5% shift (~2.4 px on 48x48)

# MLP (Model 2) pyramid defaults
MLP_HIDDEN_UNITS: tuple[int, int, int] = (512, 256, 128)
MLP_DROPOUTS: tuple[float, float, float] = (0.40, 0.35, 0.30)
MLP_L2_WEIGHT_DECAY: float = 1e-4

# ---------------------------------------------------------------------------
# Training defaults & artifact paths (used in src/train.py)
# ---------------------------------------------------------------------------
MAX_EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 8
REDUCE_LR_FACTOR: float = 0.5
REDUCE_LR_PATIENCE: int = 3
REDUCE_LR_MIN_LR: float = 1e-6

BEST_MODEL_PATH: Path = MODELS_DIR / "fer2013_cnn.keras"
LAST_MODEL_PATH: Path = MODELS_DIR / "fer2013_cnn_last.keras"
EXP2_BEST_MODEL_PATH: Path = MODELS_DIR / "fer2013_cnn_exp2.keras"
EXP2_LAST_MODEL_PATH: Path = MODELS_DIR / "fer2013_cnn_exp2_last.keras"
TRAINING_HISTORY_CSV: Path = REPORTS_DIR / "training_history.csv"
TRAINING_HISTORY_JSON: Path = REPORTS_DIR / "training_history.json"
TRAINING_REPORT_MD: Path = REPORTS_DIR / "training_report.md"


@dataclass(frozen=True)
class AugmentationConfig:
    """Configuration for light geometric augmentation applied ONLY to training data."""

    horizontal_flip_prob: float = 0.50
    rotation_range_deg: float = 10.0  # Uniform[-10, +10] degrees
    zoom_range: float = 0.10  # Uniform[0.90, 1.10] scale factor

    def __post_init__(self) -> None:
        if not (0.0 <= self.horizontal_flip_prob <= 1.0):
            raise ValueError(f"horizontal_flip_prob must be in [0, 1], got {self.horizontal_flip_prob}")
        if self.rotation_range_deg < 0.0:
            raise ValueError(f"rotation_range_deg must be >= 0, got {self.rotation_range_deg}")
        if not (0.0 <= self.zoom_range < 0.5):
            raise ValueError(f"zoom_range must be in [0, 0.5), got {self.zoom_range}")


DEFAULT_AUGMENTATION_CONFIG = AugmentationConfig()
