"""
ctgan_model.py — Train and sample from a CTGAN model.
"""

import pandas as pd
from ctgan import CTGAN

import config


def train_ctgan(train_data: pd.DataFrame, discrete_columns: list[str]) -> CTGAN:
    """Train a CTGAN model on the given data."""
    model = CTGAN(
        epochs=config.CTGAN_EPOCHS,
        batch_size=config.CTGAN_BATCH_SIZE,
        generator_dim=config.CTGAN_GENERATOR_DIM,
        discriminator_dim=config.CTGAN_DISCRIMINATOR_DIM,
        generator_lr=config.CTGAN_GENERATOR_LR,
        discriminator_lr=config.CTGAN_DISCRIMINATOR_LR,
        discriminator_steps=config.CTGAN_DISCRIMINATOR_STEPS,
        log_frequency=config.CTGAN_LOG_FREQUENCY,
        pac=config.CTGAN_PAC,
        verbose=True,
    )
    model.fit(train_data, discrete_columns)
    return model


def sample_ctgan(model: CTGAN, n_rows: int) -> pd.DataFrame:
    """Generate synthetic rows from a trained CTGAN."""
    return model.sample(n_rows)


def save_ctgan(model: CTGAN, path=None):
    """Persist the CTGAN model to disk."""
    path = path or (config.MODEL_SAVE_DIR / "ctgan.pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    return path
