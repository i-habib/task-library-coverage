#!/usr/bin/env python3
"""Save and print the verified Transformer--GRU comparison used in the paper."""

import json
from pathlib import Path

import audit_fmts_architecture_baseline as audit


if __name__ == "__main__":
    result = audit.evaluate_replicate(1, audit.load_prompt_bank())
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path("artifacts/fmts-architecture-baseline/audit/r1_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(payload)
    print(payload, end="")
