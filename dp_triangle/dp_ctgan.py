"""DP-CTGAN implementation for Direction 3."""

from __future__ import annotations

import json
import math
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import config

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)


def print_device_banner(device: torch.device) -> None:
    """
    Print the required Colab device banner.

    Inputs: torch device.
    Outputs: device and GPU summary to stdout.
    Lifecycle stage: Stage 2 - Generative Model Training.
    Reference: Colab execution constraint from user specification.
    """
    print(f"[Device] Using: {device}")
    if torch.cuda.is_available():
        print(f"[GPU] {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("[GPU] No CUDA device found - running on CPU (expect slow training)")


class Generator(nn.Module):
    """Generator network for tabular DP-CTGAN."""

    def __init__(self, noise_dim: int, data_dim: int) -> None:
        """
        Initialize the generator network.

        Inputs: latent noise dimension and flattened data dimension.
        Outputs: initialized generator module.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: user-specified Direction 3 architecture.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, config.DP_GENERATOR_DIMS[0]),
            nn.BatchNorm1d(config.DP_GENERATOR_DIMS[0]),
            nn.ReLU(),
            nn.Linear(config.DP_GENERATOR_DIMS[0], config.DP_GENERATOR_DIMS[1]),
            nn.BatchNorm1d(config.DP_GENERATOR_DIMS[1]),
            nn.ReLU(),
            nn.Linear(config.DP_GENERATOR_DIMS[1], data_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the generator.

        Inputs: latent noise tensor.
        Outputs: synthetic feature tensor in [-1, 1].
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: standard GAN forward pass.
        """
        return self.model(noise)


class Discriminator(nn.Module):
    """Discriminator network compatible with Opacus."""

    def __init__(self, data_dim: int) -> None:
        """
        Initialize the discriminator network.

        Inputs: flattened data dimension.
        Outputs: initialized discriminator module.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: user-specified Direction 3 architecture with no BatchNorm in discriminator.
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, config.DP_DISC_DIMS[0]),
            nn.LeakyReLU(0.2),
            nn.Linear(config.DP_DISC_DIMS[0], config.DP_DISC_DIMS[1]),
            nn.LeakyReLU(0.2),
            nn.Linear(config.DP_DISC_DIMS[1], 1),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass through the discriminator.

        Inputs: encoded feature batch.
        Outputs: discriminator logits.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: standard GAN forward pass.
        """
        return self.model(batch)


class DPCTGANSynthesizer:
    """Differentially private CTGAN-style synthesizer using Opacus on the discriminator."""

    def __init__(
        self,
        epsilon: float | None,
        epochs: int = 300,
        batch_size: int = 500,
        noise_dim: int = 128,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        random_seed: int = 42,
    ) -> None:
        """
        Configure the synthesizer.

        Inputs: privacy budget and GAN hyperparameters.
        Outputs: configured synthesizer ready for fitting.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Yousefpour et al. (2021) Opacus arXiv:2109.12298.
        """
        self.epsilon = epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.random_seed = random_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.generator: Generator | None = None
        self.discriminator: nn.Module | None = None
        self.ordinal_encoder: OrdinalEncoder | None = None
        self.scaler: MinMaxScaler | None = None
        self.column_order: list[str] = []
        self.categorical_cols: list[str] = []
        self.numerical_cols: list[str] = []
        self.original_dtypes: dict[str, str] = {}
        self.categorical_cardinality: dict[str, int] = {}
        self.data_dim = 0
        self.fitted = False
        self.diverged = False
        self.epsilon_actual: float | None = None
        self.final_g_loss: float | None = None
        self.final_d_loss: float | None = None
        self.epochs_completed = 0
        self._fallback_train_df: pd.DataFrame | None = None
        self._privacy_engine = None

    def _seed_everything(self) -> None:
        """
        Set deterministic seeds.

        Inputs: none.
        Outputs: seeded torch, numpy, and random modules.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: project reproducibility requirement.
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def _metadata_paths(self) -> dict[str, Path]:
        """
        Build metadata output paths.

        Inputs: none.
        Outputs: metadata artifact paths under models/saved.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Direction 3 preprocessing persistence specification.
        """
        return {
            "encoder": config.MODEL_SAVE_DIR / "dp_encoder.pkl",
            "scaler": config.MODEL_SAVE_DIR / "dp_scaler.pkl",
            "metadata": config.MODEL_SAVE_DIR / "dp_metadata.json",
        }

    def _fit_preprocessors(self, train_df: pd.DataFrame) -> np.ndarray:
        """
        Fit preprocessors and encode the training dataframe.

        Inputs: training dataframe.
        Outputs: encoded and normalized matrix in [-1, 1].
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Direction 3 preprocessing specification.
        """
        self.column_order = train_df.columns.tolist()
        self.categorical_cols = train_df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.numerical_cols = [column for column in self.column_order if column not in self.categorical_cols]
        self.original_dtypes = {column: str(dtype) for column, dtype in train_df.dtypes.items()}
        self._fallback_train_df = train_df.copy()

        categorical_encoded = np.empty((len(train_df), 0), dtype=np.float32)
        if self.categorical_cols:
            self.ordinal_encoder = OrdinalEncoder()
            raw_cats = self.ordinal_encoder.fit_transform(train_df[self.categorical_cols].astype(str))
            normalized_cats = []
            for index, column in enumerate(self.categorical_cols):
                cardinality = len(self.ordinal_encoder.categories_[index])
                self.categorical_cardinality[column] = cardinality
                denom = max(cardinality - 1, 1)
                normalized_cats.append((raw_cats[:, index] / denom).reshape(-1, 1))
            categorical_encoded = np.concatenate(normalized_cats, axis=1).astype(np.float32)

        numerical_encoded = np.empty((len(train_df), 0), dtype=np.float32)
        if self.numerical_cols:
            self.scaler = MinMaxScaler()
            numerical_encoded = self.scaler.fit_transform(train_df[self.numerical_cols]).astype(np.float32)

        matrix = np.concatenate([categorical_encoded, numerical_encoded], axis=1)
        self.data_dim = matrix.shape[1]
        self._persist_preprocessors()
        return (matrix * 2.0 - 1.0).astype(np.float32)

    def _persist_preprocessors(self) -> None:
        """
        Persist preprocessing artifacts.

        Inputs: none.
        Outputs: encoder, scaler, and metadata saved to disk.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Direction 3 preprocessing persistence specification.
        """
        config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        path_map = self._metadata_paths()
        with path_map["encoder"].open("wb") as handle:
            pickle.dump(self.ordinal_encoder, handle)
        with path_map["scaler"].open("wb") as handle:
            pickle.dump(self.scaler, handle)
        metadata = {
            "column_order": self.column_order,
            "categorical_cols": self.categorical_cols,
            "numerical_cols": self.numerical_cols,
            "target_col": config.DP_TARGET_COL,
            "sensitive_col": config.DP_SENSITIVE_COL,
        }
        path_map["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def _build_models(self) -> None:
        """
        Build generator and discriminator modules.

        Inputs: none.
        Outputs: generator and discriminator modules on the active device.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Direction 3 network architecture.
        """
        self.generator = Generator(self.noise_dim, self.data_dim).to(self.device)
        self.discriminator = Discriminator(self.data_dim).to(self.device)

    def _make_loader(self, encoded_data: np.ndarray, batch_size: int) -> DataLoader:
        """
        Create the training DataLoader.

        Inputs: encoded training matrix and batch size.
        Outputs: DataLoader with uniform mini-batches.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Opacus requirement for drop_last=True.
        """
        tensor_dataset = TensorDataset(torch.tensor(encoded_data, dtype=torch.float32))
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def _optimizer_matches_module(
        self,
        optimizer: torch.optim.Optimizer,
        module: nn.Module,
    ) -> bool:
        """
        Check whether optimizer parameters match module parameters by identity.

        Inputs: optimizer and module to compare.
        Outputs: True if the optimizer targets the module's current parameters.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Opacus validates module and optimizer parameter alignment.
        """
        optimizer_param_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        module_param_ids = {id(parameter) for parameter in module.parameters()}
        return optimizer_param_ids == module_param_ids

    def _rebuild_optimizer_for_module(
        self,
        optimizer: torch.optim.Optimizer,
        module: nn.Module,
    ) -> torch.optim.Optimizer:
        """
        Recreate an optimizer bound to a replacement module.

        Inputs: source optimizer and replacement module.
        Outputs: new optimizer instance configured with the original defaults.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Opacus module replacement can invalidate optimizer parameter bindings.
        """
        optimizer_class = type(optimizer)
        return optimizer_class(module.parameters(), **optimizer.defaults)

    def _try_attach_privacy(self, optimizer_d: torch.optim.Optimizer, loader: DataLoader):
        """
        Wrap the discriminator with Opacus when epsilon is set.

        Inputs: discriminator optimizer and data loader.
        Outputs: maybe-private discriminator, optimizer, loader, and privacy engine.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Yousefpour et al. (2021) Opacus arXiv:2109.12298.
        """
        if self.epsilon is None:
            return self.discriminator, optimizer_d, loader, None

        try:
            from opacus import PrivacyEngine  # type: ignore
            from opacus.validators import ModuleValidator  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Opacus is required for DP training. Install dependencies from requirements_direction3.txt."
            ) from exc

        assert self.discriminator is not None
        fixed_discriminator = ModuleValidator.fix(self.discriminator).to(self.device)
        if not self._optimizer_matches_module(optimizer_d, fixed_discriminator):
            optimizer_d = self._rebuild_optimizer_for_module(optimizer_d, fixed_discriminator)

        privacy_engine = PrivacyEngine()
        private_module, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
            module=fixed_discriminator,
            optimizer=optimizer_d,
            data_loader=loader,
            epochs=self.epochs,
            target_epsilon=self.epsilon,
            target_delta=self.target_delta,
            max_grad_norm=self.max_grad_norm,
        )
        self.discriminator = private_module
        return private_module, private_optimizer, private_loader, privacy_engine

    def _fit_once(self, train_df: pd.DataFrame, batch_size: int) -> None:
        """
        Run a single training attempt.

        Inputs: training dataframe and batch size.
        Outputs: trained generator/discriminator state for one attempt.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: standard GAN BCE optimization with Opacus wrapping for discriminator privacy.
        """
        self._seed_everything()
        encoded_data = self._fit_preprocessors(train_df)
        self._build_models()
        assert self.generator is not None
        assert self.discriminator is not None

        loader = self._make_loader(encoded_data, batch_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=2e-4)
        optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=2e-4)
        _, optimizer_d, loader, privacy_engine = self._try_attach_privacy(optimizer_d, loader)
        self._privacy_engine = privacy_engine

        consecutive_nan_epochs = 0
        for epoch in range(1, self.epochs + 1):
            batch_g_losses: list[float] = []
            batch_d_losses: list[float] = []

            for (real_batch_cpu,) in loader:
                real_batch = real_batch_cpu.to(self.device)
                current_batch_size = real_batch.size(0)
                real_labels = torch.ones((current_batch_size, 1), device=self.device)
                fake_labels = torch.zeros((current_batch_size, 1), device=self.device)

                optimizer_d.zero_grad()
                real_logits = self.discriminator(real_batch)
                d_loss_real = criterion(real_logits, real_labels)

                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                fake_batch = self.generator(noise)
                fake_logits = self.discriminator(fake_batch.detach())
                d_loss_fake = criterion(fake_logits, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                optimizer_d.step()

                optimizer_g.zero_grad()
                noise = torch.randn(current_batch_size, self.noise_dim, device=self.device)
                generated_batch = self.generator(noise)
                generated_logits = self.discriminator(generated_batch)
                g_loss = criterion(generated_logits, real_labels)
                g_loss.backward()
                optimizer_g.step()

                batch_g_losses.append(float(g_loss.detach().cpu()))
                batch_d_losses.append(float(d_loss.detach().cpu()))

            self.final_g_loss = float(np.mean(batch_g_losses)) if batch_g_losses else math.nan
            self.final_d_loss = float(np.mean(batch_d_losses)) if batch_d_losses else math.nan
            self.epochs_completed = epoch

            if np.isnan(self.final_g_loss) or np.isnan(self.final_d_loss):
                consecutive_nan_epochs += 1
            else:
                consecutive_nan_epochs = 0

            epsilon_spent: str | float
            if privacy_engine is not None:
                epsilon_spent = float(privacy_engine.get_epsilon(self.target_delta))
                self.epsilon_actual = epsilon_spent
            else:
                epsilon_spent = "N/A"

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"[DP-CTGAN][ε={self.epsilon if self.epsilon is not None else 'None'}]"
                    f"[{epoch}/{self.epochs}] G={self.final_g_loss:.4f} "
                    f"D={self.final_d_loss:.4f} ε_spent={epsilon_spent}"
                )

            if consecutive_nan_epochs > 10:
                print(f"WARNING: Training diverged at epoch {epoch} for ε={self.epsilon}")
                self.diverged = True
                break

        self.fitted = True

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train the synthesizer.

        Inputs: Adult train dataframe.
        Outputs: trained synthesizer state.
        Lifecycle stage: Stage 2 - Generative Model Training.
        Reference: Yousefpour et al. (2021) Opacus arXiv:2109.12298.
        """
        print_device_banner(self.device)
        try:
            self._fit_once(train_df, self.batch_size)
        except torch.cuda.OutOfMemoryError:
            if self.batch_size == 256:
                raise
            print("[OOM] Reduced batch_size to 256 and retried")
            torch.cuda.empty_cache()
            self.batch_size = 256
            self._fit_once(train_df, self.batch_size)

    def _decode_tensor(self, generated: np.ndarray) -> pd.DataFrame:
        """
        Decode generated samples back to the original schema.

        Inputs: generated array in [-1, 1].
        Outputs: inverse-transformed dataframe matching training schema.
        Lifecycle stage: Stage 2 - Sampling.
        Reference: Direction 3 preprocessing inversion requirement.
        """
        zero_one = np.clip((generated + 1.0) / 2.0, 0.0, 1.0)
        decoded: dict[str, pd.Series] = {}
        cursor = 0

        for column in self.categorical_cols:
            cardinality = self.categorical_cardinality[column]
            denom = max(cardinality - 1, 1)
            raw_indices = np.round(zero_one[:, cursor] * denom).astype(int)
            raw_indices = np.clip(raw_indices, 0, cardinality - 1)
            categories = np.asarray(self.ordinal_encoder.categories_[self.categorical_cols.index(column)])
            decoded[column] = pd.Series(categories[raw_indices], dtype="object")
            cursor += 1

        if self.numerical_cols:
            numeric_count = len(self.numerical_cols)
            numeric_block = self.scaler.inverse_transform(zero_one[:, cursor : cursor + numeric_count])
            for index, column in enumerate(self.numerical_cols):
                series = pd.Series(numeric_block[:, index])
                if "int" in self.original_dtypes[column]:
                    series = series.round().astype(int)
                decoded[column] = series

        out_df = pd.DataFrame(decoded)
        out_df = out_df[self.column_order]
        if config.DP_TARGET_COL in out_df.columns and "int" in self.original_dtypes.get(config.DP_TARGET_COL, ""):
            out_df[config.DP_TARGET_COL] = out_df[config.DP_TARGET_COL].round().astype(int)
        return out_df

    def _fallback_sample(self, n: int) -> pd.DataFrame:
        """
        Produce fallback synthetic rows after training divergence.

        Inputs: number of rows to generate.
        Outputs: fallback synthetic dataframe from bootstrapped real rows.
        Lifecycle stage: Stage 2 - Sampling fallback.
        Reference: user-specified NaN-guard fallback.
        """
        assert self._fallback_train_df is not None
        sampled = self._fallback_train_df.sample(n=n, replace=True, random_state=self.random_seed).reset_index(drop=True)
        for column in self.numerical_cols:
            noise = np.random.normal(loc=0.0, scale=0.01, size=n)
            sampled[column] = sampled[column].astype(float) + noise
            if "int" in self.original_dtypes.get(column, ""):
                sampled[column] = sampled[column].round().astype(int)
        if config.DP_TARGET_COL in sampled.columns and "int" in self.original_dtypes.get(config.DP_TARGET_COL, ""):
            sampled[config.DP_TARGET_COL] = sampled[config.DP_TARGET_COL].round().astype(int)
        return sampled[self.column_order]

    def sample(self, n: int) -> pd.DataFrame:
        """
        Generate synthetic rows.

        Inputs: desired synthetic row count.
        Outputs: synthetic dataframe with original schema.
        Lifecycle stage: Stage 2 - Sampling.
        Reference: standard GAN sampling plus user-specified divergence fallback.
        """
        if self.diverged:
            print("WARNING: Returning fallback synthetic data (training diverged)")
            return self._fallback_sample(n)

        assert self.generator is not None
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n, self.noise_dim, device=self.device)
            generated = self.generator(noise).detach().cpu().numpy()
        return self._decode_tensor(generated)

    def save(self, path: Path) -> None:
        """
        Persist the synthesizer to disk.

        Inputs: output pickle path.
        Outputs: serialized synthesizer on disk.
        Lifecycle stage: Stage 2 - Persistence.
        Reference: project model persistence requirement.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: Path) -> "DPCTGANSynthesizer":
        """
        Reload a saved synthesizer.

        Inputs: saved pickle path.
        Outputs: reloaded synthesizer object.
        Lifecycle stage: Stage 2 - Persistence.
        Reference: project model persistence requirement.
        """
        with path.open("rb") as handle:
            return pickle.load(handle)
