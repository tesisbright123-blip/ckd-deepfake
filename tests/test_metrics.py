"""Unit tests for src/evaluation/metrics.py.

Run: pytest tests/test_metrics.py -v

Targets:
    compute_auc(labels, scores)
        perfect classifier -> 1.0
        random classifier  -> ~0.5
    compute_forgetting(prev_auc, curr_auc)
        drop from 0.90 -> 0.80 gives forgetting = 0.10
    compute_cde(aucs_per_gen, latency_ms, size_mb)
        higher AUC and lower latency/size -> higher CDE
    compute_cgrs(aucs_old_gens_after_update)
        all gens retained perfectly -> 1.0
"""
