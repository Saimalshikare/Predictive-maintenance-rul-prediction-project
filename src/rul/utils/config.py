from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path=DEFAULT_CONFIG):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(cfg, key):
    folder = PROJECT_ROOT / cfg["paths"][key]
    folder.mkdir(parents=True, exist_ok=True)
    return folder