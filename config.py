"""
config.py — All hyperparameters in one place.
"""

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
DATA_DIR = "data"
RESULTS_DIR = "results"
MODEL_SAVE_DIR = "models/saved"
