"""Check every file listed in MANIFEST.json against its recorded size and SHA-256.

  python verify_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
bad = 0
for rel, rec in json.loads((ROOT / "MANIFEST.json").read_text()).items():
    p = ROOT / rel
    if not p.exists():
        print(f"missing  {rel}")
        bad += 1
    elif p.stat().st_size != rec["bytes"] or hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
        print(f"mismatch {rel}")
        bad += 1
print("all files verified" if not bad else f"{bad} problem(s)")
sys.exit(1 if bad else 0)
