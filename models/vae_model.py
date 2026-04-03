"""
vae_model.py — Variational Autoencoder for tabular data synthesis.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config


class TabularVAE:
    """A simple VAE that works on mixed-type tabular data."""

    def __init__(self, continuous_cols: list[str], discrete_cols: list[str]):
        self.continuous_cols = continuous_cols
        self.discrete_cols = discrete_cols
        self.scaler = StandardScaler()
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder = None
        self.decoder = None
        self.vae = None
        self._input_dim = None

    # ── preprocessing ────────────────────────────────────────────────

    def _preprocess(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        parts = []
        if self.continuous_cols:
            cont = df[self.continuous_cols].values.astype("float32")
            if fit:
                cont = self.scaler.fit_transform(cont)
            else:
                cont = self.scaler.transform(cont)
            parts.append(cont)
        if self.discrete_cols:
            disc = df[self.discrete_cols].astype(str)
            if fit:
                disc = self.ohe.fit_transform(disc)
            else:
                disc = self.ohe.transform(disc)
            parts.append(disc.astype("float32"))
        return np.concatenate(parts, axis=1)

    def _postprocess(self, data: np.ndarray) -> pd.DataFrame:
        idx = 0
        result = {}
        if self.continuous_cols:
            n = len(self.continuous_cols)
            cont = self.scaler.inverse_transform(data[:, idx : idx + n])
            for i, col in enumerate(self.continuous_cols):
                result[col] = cont[:, i]
            idx += n
        if self.discrete_cols:
            n_ohe = len(self.ohe.get_feature_names_out())
            disc_encoded = data[:, idx : idx + n_ohe]
            disc_labels = self.ohe.inverse_transform(disc_encoded)
            for i, col in enumerate(self.discrete_cols):
                result[col] = disc_labels[:, i]
        return pd.DataFrame(result)

    # ── model building ───────────────────────────────────────────────

    def _build(self):
        latent_dim = config.VAE_LATENT_DIM
        kl_weight = config.VAE_KL_WEIGHT

        # Encoder
        enc_input = keras.Input(shape=(self._input_dim,))
        x = enc_input
        for dim in config.VAE_ENCODER_DIMS:
            x = keras.layers.Dense(dim, activation="relu")(x)
        z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)

        def sampling(args):
            mu, log_var = args
            eps = tf.random.normal(shape=tf.shape(mu))
            return mu + tf.exp(0.5 * log_var) * eps

        z = keras.layers.Lambda(sampling)([z_mean, z_log_var])
        self.encoder = keras.Model(enc_input, [z_mean, z_log_var, z], name="encoder")

        # Decoder
        dec_input = keras.Input(shape=(latent_dim,))
        x = dec_input
        for dim in config.VAE_DECODER_DIMS:
            x = keras.layers.Dense(dim, activation="relu")(x)
        dec_output = keras.layers.Dense(self._input_dim, activation="sigmoid")(x)
        self.decoder = keras.Model(dec_input, dec_output, name="decoder")

        # VAE
        outputs = self.decoder(self.encoder(enc_input)[2])
        self.vae = keras.Model(enc_input, outputs, name="vae")

        # Loss
        reconstruction_loss = tf.reduce_mean(
            keras.losses.mse(enc_input, outputs)
        ) * self._input_dim
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.vae.add_loss(reconstruction_loss + kl_weight * kl_loss)
        self.vae.compile(optimizer=keras.optimizers.Adam(config.VAE_LEARNING_RATE))

    # ── public API ───────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame):
        data = self._preprocess(df, fit=True)
        self._input_dim = data.shape[1]
        self._build()
        self.vae.fit(
            data, data,
            epochs=config.VAE_EPOCHS,
            batch_size=config.VAE_BATCH_SIZE,
            verbose=1,
        )

    def sample(self, n_rows: int) -> pd.DataFrame:
        z = np.random.normal(size=(n_rows, config.VAE_LATENT_DIM)).astype("float32")
        decoded = self.decoder.predict(z, verbose=0)
        return self._postprocess(decoded)

    def save(self, directory=None):
        directory = directory or (config.MODEL_SAVE_DIR / "vae")
        directory.mkdir(parents=True, exist_ok=True)
        self.vae.save(directory / "vae_model.keras")
        return directory
