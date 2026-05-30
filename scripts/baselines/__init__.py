"""Baseline experiments for the CKD thesis comparison table (BAB V).

Scope (per the "minimal balanced" decision):
    B1 — Ensemble teacher direct evaluation (this dir, eval-only, no GPU).
    B3 — Naive sequential fine-tuning (via scripts/05 ``--method none``).
    B4 — Continual learning without KD (via scripts/05 ``--no-soft-labels``).

Dropped (reported as estimate / future work):
    B2 — Full retrain · B5 — Static KD.

A1 (KD contribution) is answered by B4 vs the full CKD pipeline; no separate
A1 run is needed.
"""
