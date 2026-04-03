# Synthetic Data Generation Lifecycle: Project Documentation

## Project Overview

This project implements a complete pipeline for generating and evaluating synthetic data. It focuses on tabular data generation using state-of-the-art Deep Learning models. The primary objective is to produce synthetic datasets that closely mimic a real dataset (in this case, the UCI Adult Income dataset) while quantifying the trade-offs between **Data Utility**, **Privacy**, and **Fairness**.

The pipeline is entirely automated and designed in a modular fashion, allowing for isolated execution of its different stages. The main orchestrator is `main.py`, which systematically runs through data preprocessing, model training, evaluation, and visualization.

---

## The 4-Stage Lifecycle

The system operates in four distinct stages:

### Stage 1: Data Acquisition & Preprocessing (`data/loader.py`)

- **Data Source**: Fetches the Adult Income dataset from the UC Irvine Machine Learning Repository.
- **Cleaning**: Handles missing values (e.g., replacing "?" with `NaN` and dropping incomplete rows).
- **Encoding**: Converts the target variable (`income`) into a binary format (0 for `<=50K`, 1 for `>50K`).
- **Splitting**: Divides the data into training and hold-out testing sets using stratified sampling to maintain class distributions, ensuring the evaluation reflects real-world model deployment.

### Stage 2: Generative Model Training (`models/train_models.py`)

- Trains two dominant tabular generative models on the real training data:
  1. **CTGAN** (Conditional Tabular Generative Adversarial Network)
  2. **TVAE** (Tabular Variational Autoencoder)
- After training, the models are used to synthesize new datasets (`ctgan_synthetic.csv` and `tvae_synthetic.csv`), which are verified for basic validity (e.g., correct shape, bounds, no `NaN` values).

### Stage 3: Privacy & Fairness Evaluation (`evaluation/privacy_fairness.py`)

Quantifies the risks associated with the generated synthetic data:

- **Privacy check**: Tests if an attacker can confidently determine whether a specific individual was part of the original training data.
- **Fairness check**: Checks if the synthetic data inherits or exacerbates biases present in the original dataset regarding a sensitive attribute (e.g., `sex`).

### Stage 4: Utility Evaluation & Dashboard (`evaluation/metrics.py`)

Assesses how useful the synthetic data is for downstream machine learning tasks and statistical analysis:

- Calculates explicit utility scores.
- Compiles the final evaluation dashboard summarizing Utility, Privacy, and Fairness metrics into a single table (`final_evaluation_table.csv`).

Additionally, `evaluation/visualize.py` generates visual plots (distribution comparisons, correlations) to aid qualitative assessment.

---

## Underlying Concepts & Terminology

To thoroughly understand the evaluation results of this project, it is essential to comprehend the generative models used and the metrics calculated.

### 1. Generative Models

- **CTGAN (Conditional Tabular GAN)**: Standard GANs struggle with tabular data due to non-Gaussian distributions and implicit relationships between continuous and categorical columns. CTGAN introduces conditional generators and training by sampling to successfully model discrete columns and multimodal continuous distributions.
- **TVAE (Tabular Variational Autoencoder)**: An adaptation of the Variational Autoencoder designed specifically for tabular data. It learns a compressed, latent representation of the tabular data space and generates new records by sampling from this latent space and decoding them back into tabular features. TVAE often exhibits higher stability during training compared to GANs.

### 2. Utility Metrics

- **TSTR (Train-on-Synthetic, Test-on-Real) Accuracy**: A practical measure of machine learning utility. A classifier (e.g., Random Forest) is trained *exclusively* on the synthetic dataset, and its performance (accuracy) is queried against the hold-out *real* test dataset. High TSTR accuracy implies the synthetic data captured the feature-target relationships well enough to train reliable downstream models.
- **Jensen-Shannon (JS) Divergence**: A statistical metric that measures the distance between two probability distributions (the real data distribution vs. the synthetic data distribution). For numerical columns, a lower JS divergence (closer to 0) indicates that the synthetic data accurately mimics the statistical distribution of the original dataset.

### 3. Privacy Metric

- **Membership Inference Attack (MIA) Advantage**: In an MIA, a malicious actor tries to guess whether a particular data record was used in the model's training set. The "Advantage" is calculated as the attacker's accuracy minus 0.5 (random guessing base rate).
  - A value close to **0.0** indicates strong privacy (the synthetic data does not memorize original rows).
  - A value approaching **0.5** indicates weak privacy (the model has heavily memorized the training data and leaks its presence).

### 4. Fairness Metric

- **Demographic Parity Difference**: A measure of algorithmic fairness. It calculates the absolute difference in the positive outcome rate (`income > 50K`) between two demographic groups defined by a sensitive attribute (in this case, `sex`: Male vs. Female).
  - Formula: $|P(\text{income } > \text{ 50K } | \text{ Male}) - P(\text{income } > \text{ 50K } | \text{ Female})|$
  - A value of **0** indicates perfect demographic parity (the outcome is independent of the sensitive attribute). Larger values indicate disparity. We evaluate this on the synthetic data to ensure the generation process hasn't amplified existing historical biases.

---

## Configuration & Hyperparameters

The entire lifecycle is modularly configured via `config.py`. Key parameters found here include:

- **Dataset Properties**: Random seeds, test split sizes.
- **Model Hyperparameters**: Epochs, batch sizes, learning rates, dimensions for CTGAN and TVAE.
- **Evaluation Targets**: The target label (`income`), sensitive feature for fairness (`sex`), and classifier type for TSTR evaluation.

## Running the Pipeline

To execute the entire lifecycle from start to finish, run:

```bash
python main.py
```

Because the pipeline uses state-caching (checking if output files already exist), you can effortlessly re-run `main.py` to pick up where it left off, or delete specific files in the `results/` or `models/` directories to force re-execution of a specific stage.
