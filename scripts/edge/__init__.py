"""Edge-deployment helper scripts (MacBook M2 Max workflow).

Modules:
    setup_macbook_mirror  : Stage local data + rewrite CSV paths for MacBook.
    run_edge_eval_macbook : End-to-end orchestrator for 9 ckpt x 4 format
                            conversion + AUC + latency benchmarks.

Not part of the Colab/A100 training pipeline — used only for the
on-device edge benchmarking step (Step 7 in the thesis pipeline).
"""
