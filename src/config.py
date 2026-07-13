"""Loads config.yaml once and exposes it as a plain dict.

Every other module reads settings through `get_config()` instead of opening
the YAML file itself, so there is exactly one place that knows the file's
path and layout.
"""

from pathlib import Path
from functools import lru_cache

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
