"""Проверка, что окружение курса собрано правильно.

    python tools/check_env.py
"""

from __future__ import annotations

import platform
import sys

CORE = [
    ("numpy", "1.26"),
    ("pandas", "2.1"),
    ("scipy", "1.11"),
    ("sklearn", "1.4"),
    ("matplotlib", "3.8"),
]
LATER = ["seaborn", "catboost", "lightgbm", "optuna", "datasketch"]


def version_of(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "?")


def as_tuple(v: str) -> tuple:
    parts = []
    for chunk in v.split(".")[:3]:
        digits = "".join(ch for ch in chunk if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def main() -> int:
    print(f"python {platform.python_version()}  ({sys.executable})")
    if sys.version_info < (3, 10):
        print("  ! нужен python 3.10 или новее")

    problems = 0
    print("\nобязательные пакеты:")
    for name, minimum in CORE:
        got = version_of(name)
        if got is None:
            print(f"  MISSING  {name}  (нужен >= {minimum})")
            problems += 1
        elif got != "?" and as_tuple(got) < as_tuple(minimum):
            print(f"  OLD      {name} {got}  (нужен >= {minimum})")
            problems += 1
        else:
            print(f"  ok       {name} {got}")

    print("\nпонадобятся позже по курсу:")
    for name in LATER:
        got = version_of(name)
        print(f"  {'ok      ' if got else 'нет пока'} {name} {got or ''}".rstrip())

    if problems:
        print(f"\n{problems} проблем(ы). Установка: pip install -r requirements.txt")
        return 1
    print("\nОкружение готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
