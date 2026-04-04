"""
config.py — All hyperparameters in one place.
"""

from pathlib import Path

# ── Dataset ──────────────────────────────────────────────────────────
DATASET_NAME = "adult"  # UCI Adult Income dataset
UCI_DATASET_ID = 2       # ucimlrepo ID for Adult
TEST_SIZE = 0.2
RANDOM_SEED = 42

# ── CTGAN ────────────────────────────────────────────────────────────
CTGAN_EPOCHS = 300
CTGAN_BATCH_SIZE = 500
CTGAN_GENERATOR_DIM = (256, 256)
CTGAN_DISCRIMINATOR_DIM = (256, 256)
CTGAN_GENERATOR_LR = 2e-4
CTGAN_DISCRIMINATOR_LR = 2e-4
CTGAN_DISCRIMINATOR_STEPS = 1
CTGAN_LOG_FREQUENCY = True
CTGAN_PAC = 10

# ── VAE ──────────────────────────────────────────────────────────────
VAE_LATENT_DIM = 16
VAE_ENCODER_DIMS = [128, 64]
VAE_DECODER_DIMS = [64, 128]
VAE_EPOCHS = 100
VAE_BATCH_SIZE = 256
VAE_LEARNING_RATE = 1e-3
VAE_KL_WEIGHT = 1.0

# ── Differential Privacy ─────────────────────────────────────────────
DP_ENABLED = False
DP_NOISE_MULTIPLIER = 1.1
DP_L2_NORM_CLIP = 1.0
DP_MICROBATCHES = 1
DP_DELTA = 1e-5

# ── Evaluation ───────────────────────────────────────────────────────
N_SYNTHETIC_ROWS = None  # None = match real dataset size
CLASSIFIER_TYPE = "logistic_regression"  # for utility evaluation
FAIRNESS_SENSITIVE_FEATURE = "sex"
FAIRNESS_TARGET = "income"

# ── Paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_SAVE_DIR = PROJECT_ROOT / "models" / "saved"

# ── Direction 3: Privacy–Fairness–Fidelity Triangle ─────────────────
DP_EPSILON_VALUES = {
    "no_dp": None,
    "eps_10": 10.0,
    "eps_1": 1.0,
    "eps_0_5": 0.5,
    "eps_0_1": 0.1,
}
DP_EPOCHS = 300
DP_BATCH_SIZE = 500
DP_NOISE_DIM = 128
DP_TARGET_DELTA = 1e-5
DP_MAX_GRAD_NORM = 1.0
DP_N_SYNTHETIC = 10_000
DP_GENERATOR_DIMS = [256, 256]
DP_DISC_DIMS = [256, 256]
DP_SENSITIVE_COL = "sex"
DP_TARGET_COL = "income"
DP_MINORITY_CLASS = 1
