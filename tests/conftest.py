import os
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip all tests when SKIP_ALL_TESTS env var is set.

    Set SKIP_ALL_TESTS=1 (or true/yes) to mark all collected tests as skipped.
    """
    val = str(os.getenv("SKIP_ALL_TESTS", "0")).lower()
    if val in ("1", "true", "yes"):
        skip_all = pytest.mark.skip(reason="Skipped via SKIP_ALL_TESTS env var")
        for item in items:
            item.add_marker(skip_all)

