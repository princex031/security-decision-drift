"""
Machine-readable experiment output utilities.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
) -> None:
    """Write experiment rows as a CSV file."""

    rows = list(rows)

    if not rows:
        raise ValueError("Cannot write an empty result set.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    """Write run metadata as formatted JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")