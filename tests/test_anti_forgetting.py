"""Unit tests for src/training/anti_forgetting/*.

Run: pytest tests/test_anti_forgetting.py -v

Targets:
    EWC (Elastic Weight Consolidation):
        Fisher information matrix has same keys as model.named_parameters()
        penalty is zero at theta == theta_star, positive elsewhere
    LwF (Learning without Forgetting):
        distillation output matches old model on fresh inputs
        penalty uses temperature T=2.0
    Replay buffer:
        herding selection returns exactly budget-many exemplars
        per-class balance within 1 sample
        buffer.sample() returns tensors of correct shape
"""
