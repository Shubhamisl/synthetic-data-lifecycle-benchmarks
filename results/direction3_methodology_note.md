# Direction 3 Methodology Note

## Why The Raw Triangle Score Is Not Enough

The raw Privacy-Fairness-Fidelity triangle score averages:

- privacy score = `1 - MIA_Advantage`
- utility score = `TSTR / 100`
- fairness score = `1 - Demographic_Parity`

This is useful as a first-order summary, but it can become misleading when a synthetic generator collapses the positive or minority class. In that case, demographic parity may look artificially strong because both groups receive the same prediction outcome for the wrong reason: the model has stopped representing the positive class at all.

## Collapse-Aware Adjustment

Direction 3 therefore uses an adjusted reporting metric:

- `Synthetic_Positive_Rate`
- `Positive_Class_Retention = synthetic_positive_rate / real_positive_rate`
- `Collapsed_Minority_Class`
- `Collapse_Reason`
- `Triangle_Score_Adjusted = Triangle_Score * min(Positive_Class_Retention, 1.0)`

This adjustment preserves the original triangle logic for usable variants, while preventing degenerate fairness wins from being rewarded when the positive class disappears.

## Reporting Rule

For Direction 3 interpretation and paper writing:

- use `Triangle_Score_Adjusted` as the primary ranking metric
- use the raw `Triangle_Score` only as a diagnostic quantity
- treat any variant with `Collapsed_Minority_Class = True` as a failure regime, not a valid fairness improvement

## Practical Consequence In This Project

In the current Adult run, `ε=0.1` achieves raw demographic parity of `0.000`, but this occurs because the synthetic dataset collapses to `income = 0` for all generated rows. The adjusted triangle score correctly reduces this variant to `0.000` and prevents it from being misreported as the best privacy-fairness-fidelity trade-off.
