#!/usr/bin/env python3
"""Combine period logs for one evaluation configuration and summarize each arm."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.evaluation_metrics import summarize_delivery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                        help="arms.jsonl files from the same evaluation configuration")
    parser.add_argument("--out", type=Path, required=True, help="output summary.json")
    args = parser.parse_args()
    grouped = defaultdict(list)
    try:
        for path in args.input:
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("arm"), str) or not row["arm"]:
                        raise ValueError(f"{path}:{line_number}: expected a period record with an arm")
                    grouped[row["arm"]].append(row)
        report = {arm: summarize_delivery(rows) for arm, rows in sorted(grouped.items())}
        encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
