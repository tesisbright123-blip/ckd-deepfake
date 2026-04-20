"""Shared pytest fixtures for CKD test suite.

Auto-discovered by pytest. Provides in-memory fixtures only (no file I/O,
no network). Tests that need disk should use the tmp_path fixture from pytest.

Consumed by: tests/test_dataset.py, tests/test_losses.py,
             tests/test_anti_forgetting.py, tests/test_metrics.py
"""
