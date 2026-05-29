"""Extract SMPL_NEUTRAL.pkl from hmr2_data.tar.gz."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAR = ROOT / "models" / "hmr2_data.tar.gz"
OUT = ROOT / "models" / "smpl" / "SMPL_NEUTRAL.pkl"

MEMBERS = [
    "hmr2_data/data/smpl/SMPL_NEUTRAL.pkl",
    "data/smpl/SMPL_NEUTRAL.pkl",
    "SMPL_NEUTRAL.pkl",
]


def extract() -> Path:
    if not TAR.exists():
        raise FileNotFoundError(f"Missing archive: {TAR}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TAR, "r:gz") as tf:
        names = tf.getnames()
        for m in MEMBERS:
            if m in names:
                tf.extract(m, ROOT / "models" / "_tmp")
                src = ROOT / "models" / "_tmp" / m
                shutil.copy2(src, OUT)
                shutil.rmtree(ROOT / "models" / "_tmp", ignore_errors=True)
                print("extracted", OUT, OUT.stat().st_size)
                return OUT
        # fuzzy search
        for n in names:
            if n.endswith("SMPL_NEUTRAL.pkl"):
                tf.extract(n, ROOT / "models" / "_tmp")
                src = ROOT / "models" / "_tmp" / n
                shutil.copy2(src, OUT)
                shutil.rmtree(ROOT / "models" / "_tmp", ignore_errors=True)
                print("extracted", OUT, OUT.stat().st_size)
                return OUT
    raise FileNotFoundError("SMPL_NEUTRAL.pkl not found inside archive")


if __name__ == "__main__":
    extract()
