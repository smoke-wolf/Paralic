"""Unit tests of the DOM-free browser modules (web/js/motion.js, position.js), run with node."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_browser_modules_with_node():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    files = sorted(str(f) for f in (ROOT / "tests" / "js").glob("*.test.mjs"))
    result = subprocess.run([node, "--test", *files], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
