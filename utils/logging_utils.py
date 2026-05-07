import csv
import os
import time
from typing import Iterable, Mapping


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def append_csv(path: str, fieldnames: Iterable[str], row: Mapping[str, object]) -> None:
    ensure_parent_dir(path)
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    last_error = None
    for attempt in range(10):
        try:
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.5 * (attempt + 1))
    raise last_error
