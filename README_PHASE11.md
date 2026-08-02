# Phase 11 — Defect and localization hard-example iteration

Phase 11 targets the remaining errors of the promoted 10,000-example pilot.
The Phase 10 adapter is the frozen comparison point; its weights and raw
predictions remain external artifacts, while its compact results are tracked in
[`docs/results/phase10/pilot_results.json`](docs/results/phase10/pilot_results.json).

## Evidence

The pilot reduced the complete held-out benchmark failure rate from 53.86% to
46.00%. Product identification reached 98.67%, abstention reached 99.33%, and
defect-identification F1 improved from 0.1333 to 0.3210. The largest remaining
failure groups are wrong defect (326), incomplete or incorrect facts (330), and
wrong or adjacent location (265 combined).

## Development sequence

1. Build a deterministic failure-analysis command over evaluation records.
2. Produce defect confusion tables by category and identify label aliases,
   compound-label partial matches, and visually close defect pairs.
3. Produce exact and adjacent localization confusion tables by category and
   anomaly size.
4. Audit the single invalid structured report and add a bounded JSON recovery
   path if the failure is syntactic rather than semantic.
5. Generate a reproducible hard-example training manifest without drawing from
   validation or held-out benchmark records.
6. Run CPU tests and a Colab one-batch gate before any additional training.
7. Train a separately named adapter and compare it with the frozen Phase 10
   result on exactly the same validation and test records.

## Promotion gates

- held-out failure rate below 46%;
- defect-identification F1 above 0.321;
- evidence fact coverage above 0.474;
- exact localization accuracy above 0.467 without reducing adjacent-tolerance
  accuracy below 0.863;
- product accuracy at least 0.98;
- structured JSON and schema validity of 1.0;
- abstention accuracy at least 0.99;
- zero unsupported root-cause and safety claims.

No Phase 11 training begins until the failure analysis and hard-example
selection are reproducible and leakage checks pass.
