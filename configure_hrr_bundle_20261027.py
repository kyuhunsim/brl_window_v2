"""Resolve packaged simulator paths after extraction; no Real communication."""
import hashlib
import json
from pathlib import Path
import yaml

root = Path(__file__).resolve().parent
manifest = json.loads((root / "hrr_manifest_20261027.json").read_text())
for item in manifest["models"]:
    folder = root / "pneu_rl/models" / item["name"]
    for filename, expected in item["files_sha256"].items():
        if filename == "cfg.yaml":
            continue  # This file intentionally changes its local library path.
        actual = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {folder / filename}")
    cfg_path = folder / "cfg.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["obs"]["library_path"] = str((folder / "simulator.so").resolve())
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    print(f"Ready: {item['name']} (epi2000, {item['type']})")
