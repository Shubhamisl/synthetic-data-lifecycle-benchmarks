## Direction 3: dp_triangle/

Entry point:
  python -m dp_triangle.run_direction3 --dry-run   # validate first
  python -m dp_triangle.run_direction3              # full run (Colab)

Key constraints:
- Never hardcode device="cpu". All tensors use .to(device).
- Opacus wraps discriminator only, not generator.
- Discriminator must not use BatchNorm — use LayerNorm or none.
- DataLoader must use drop_last=True for DP accounting.
- All randomness seeded via RANDOM_SEED from config.py.
- Caching: check output files before recomputing any stage.

Do not modify:
- evaluation/metrics.py
- evaluation/privacy_fairness.py
- main.py
- models/train_models.py
