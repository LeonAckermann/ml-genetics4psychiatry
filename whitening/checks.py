"""The check object every diagnostic returns.

A check is a named verdict with the numbers that produced it.  ``status`` is
one of:

* ``ok``   – nothing to do.
* ``warn`` – the transform will still run, but a modelling choice was made for
  you (imputed zeros, a rescaled diagonal, a clipped eigenvalue) and it belongs
  in the write-up.
* ``fail`` – whitening by this matrix would produce numbers you cannot defend.
* ``skip`` – the check needs an input that was not supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

STATUS_ORDER = {"ok": 0, "skip": 1, "warn": 2, "fail": 3}
STATUS_MARK = {"ok": "PASS", "skip": "SKIP", "warn": "WARN", "fail": "FAIL"}


@dataclass
class Check:
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "details": _jsonable(self.details),
        }


def worst(checks: list[Check]) -> str:
    return max((c.status for c in checks), key=lambda s: STATUS_ORDER[s], default="ok")


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (v != v or v in (float("inf"), float("-inf"))) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj
