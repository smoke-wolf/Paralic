import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow end-to-end tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow test: use --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
