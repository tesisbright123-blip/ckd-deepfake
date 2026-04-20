"""Unit tests for src/training/losses.py.

Run: pytest tests/test_losses.py -v

Targets:
    DistillationLoss (KL divergence with temperature T=4.0)
    CrossEntropyLoss wrapper
    ContinualLoss: L = alpha*L_KD + beta*L_retention + gamma*L_CE
        asserts alpha + beta + gamma == 1.0
        asserts loss is non-negative
        asserts gradient flows through student logits
"""
