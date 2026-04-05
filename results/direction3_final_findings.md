# Direction 3 Final Findings

## Adult Triangle Summary

The Adult privacy-fairness-fidelity experiment completed successfully across all five epsilon variants after adding collapse-aware evaluation.

## Final Interpretation

- `ε=0.1` is a collapse regime, not a genuine fairness success.
- The positive class disappears entirely at `ε=0.1`, producing an artificial demographic parity value of `0.000`.
- The adjusted triangle score is the correct reporting metric for Direction 3 because it penalizes positive-class erasure.
- Under the adjusted score, `ε=10.0` is the best usable privacy-fairness-fidelity trade-off in the current run.

## Research Meaning

These results support a stronger and more defensible conclusion than the raw dashboard alone:

- stronger privacy can damage subgroup utility severely
- apparent fairness gains under very low epsilon may reflect collapse rather than equitable representation
- subgroup-aware and collapse-aware reporting is necessary for trustworthy synthetic data evaluation

## How To Use This In The Paper

- present Adult as the flagship triangle case study
- discuss `ε=0.1` as a boundary or failure regime
- report `Triangle_Score_Adjusted` as the primary ranking metric
- use `ε=10.0` as the best usable trade-off in the current experiment, not `ε=0.1`
