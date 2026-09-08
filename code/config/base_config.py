"""Project paths configured through environment variables.

The defaults keep inputs and outputs outside the importable ``code`` package.
Importing this module never creates directories or files.
"""

from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


_DEFAULT_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIR = _path_from_env("URBANFRINGEPV_ROOT", _DEFAULT_ROOT)
INPUT_DIR = _path_from_env("URBANFRINGEPV_INPUT_DIR", ROOT_DIR / "input_data")
OUTPUT_DIR = _path_from_env("URBANFRINGEPV_OUTPUT_DIR", ROOT_DIR / "output")
