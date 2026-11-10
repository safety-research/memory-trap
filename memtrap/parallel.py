"""Thread-pool map."""

from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def parallel_map(fn: Callable, items, workers: int = 1) -> list:
    """Map `fn` over `items` in order. `fn` must catch its own errors."""
    items = list(items)
    if workers and workers > 1 and len(items) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(fn, items))
    return [fn(x) for x in items]
