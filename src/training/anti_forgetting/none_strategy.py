"""No-op anti-forgetting strategy for the naive fine-tuning baseline (B3).

Naive sequential fine-tuning means: continue training on the new generation
WITHOUT any catastrophic-forgetting protection — no EWC penalty, no replay
buffer, no LwF retention KD. The model is simply fine-tuned from the previous
checkpoint on the new data.

This is intentionally a thin subclass of :class:`AntiForgettingStrategy` whose
hooks are all the base-class no-ops. Having a named concrete class (rather than
instantiating the abstract base directly) keeps logs/checkpoint paths readable
(``gen2_none``, ``gen3_none``) and makes the dispatch in
``scripts/05_continual_distillation.py`` symmetric with the other methods.

Because ``provides_retention_logits`` is False, ``ContinualDistillationLoss``
collapses to ``alpha*KD + gamma*CE`` (renormalised) when soft labels are present
— i.e. naive fine-tuning still distills from the teacher ensemble; it only drops
the anti-forgetting machinery. That matches the B3 baseline definition: "training
configuration identical to S1/S2/S3 except without EWC and without replay".

Called by:
    scripts/05_continual_distillation.py (``--method none``)
Reads / Writes: none.
"""
from __future__ import annotations

from src.training.anti_forgetting.base import AntiForgettingStrategy


class NoOpStrategy(AntiForgettingStrategy):
    """Naive fine-tuning: no anti-forgetting protection at all.

    All lifecycle hooks inherit the base class's safe no-op defaults:
    - ``before_training`` does nothing (no buffer, no Fisher).
    - ``penalty`` returns zero (no EWC term).
    - ``previous_logits`` returns None (no LwF retention KD).
    - ``augment_dataloader`` passes the loader through unchanged (no replay).
    """

    name = "none"
    provides_retention_logits = False
