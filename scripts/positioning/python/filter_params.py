# -*- coding: utf-8 -*-
"""
filter_params.py — Central parameter store for all filter × mode × dataset
combinations.

Parameters are persisted in filter_params.json (auto-created and updated by
ins_genetic.py and ins_genetic_fast.py).  This module provides a clean Python
API to read and write them.

Structure of filter_params.json:
    {
      "<filter_name>": {
        "2d": { "<dataset_name>": { ...params... }, ... },
        "3d": { "<dataset_name>": { ...params... }, ... }
      },
      ...
    }

Usage (read):
    import filter_params
    p = filter_params.get('esekfs_enhanced', mode_3d=True, dataset='10_03_0027')
    # Returns the params dict, or None if not yet tuned.

Usage (write — called automatically by optimisers):
    filter_params.set('esekfs_enhanced', mode_3d=True, dataset='10_03_0027',
                      params={...}, cost=12.3)

Usage (summary):
    filter_params.print_summary()
"""
import json
from pathlib import Path

_STORE = Path(__file__).parent / 'filter_params.json'


def _load() -> dict:
    """Load the JSON store.

    A MISSING file is the legitimate "nothing tuned yet" case → empty dict.
    A file that EXISTS but fails to parse (truncated/half-written by an
    interrupted optimiser, merge conflict, corruption) must NOT be silently
    treated as "no tuning": that would make every filter fall back to default
    parameters with no warning — the exact silent-degradation this project
    forbids. Fail loudly instead so the corrupt store is fixed, not ignored.
    """
    if not _STORE.exists():
        return {}
    try:
        return json.loads(_STORE.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"filter_params store {_STORE} exists but is not valid JSON "
            f"({e}). Refusing to silently fall back to default parameters for "
            f"every filter. Restore or regenerate the file (ins_genetic.py)."
        ) from e


def _save(data: dict) -> None:
    _STORE.write_text(json.dumps(data, indent=2), encoding='utf-8')


def get(filter_name: str, mode_3d: bool, dataset: str) -> dict | None:
    """
    Return the best known parameter dict for (filter, mode, dataset), or
    None if this combination has not been tuned yet.
    """
    mode = '3d' if mode_3d else '2d'
    data = _load()
    entry = data.get(filter_name, {}).get(mode, {}).get(dataset)
    if entry is None:
        return None
    return entry.get('params')


def get_cost(filter_name: str, mode_3d: bool, dataset: str) -> float | None:
    """Return the optimisation cost recorded for this combination, or None."""
    mode = '3d' if mode_3d else '2d'
    data = _load()
    entry = data.get(filter_name, {}).get(mode, {}).get(dataset)
    if entry is None:
        return None
    return entry.get('cost')


def set(filter_name: str, mode_3d: bool, dataset: str,
        params: dict, cost: float = None, metadata: dict = None) -> None:
    """
    Save (or overwrite) the parameter dict for (filter, mode, dataset).

    Args:
        filter_name : e.g. 'esekfs_enhanced'
        mode_3d     : True = 3D, False = 2D
        dataset     : e.g. '10_03_0027'
        params      : the parameter dict to store
        cost        : optimisation cost (lower = better)
        metadata    : optional extra info (evaluations, date, …)
    """
    mode = '3d' if mode_3d else '2d'
    data = _load()
    data.setdefault(filter_name, {}).setdefault(mode, {})[dataset] = {
        'params':   {k: float(v) for k, v in params.items()},
        'cost':     float(cost) if cost is not None else None,
        'metadata': metadata or {},
    }
    _save(data)


def all_keys() -> list[tuple]:
    """Return list of (filter_name, mode, dataset) tuples present in the store."""
    keys = []
    for fname, modes in _load().items():
        for mode, datasets in modes.items():
            for ds in datasets:
                keys.append((fname, mode, ds))
    return keys


def print_summary() -> None:
    """Print a human-readable table of all stored parameters and their costs."""
    data = _load()
    if not data:
        print("filter_params.json is empty — run ins_genetic_fast.py first.")
        return

    print()
    print("=" * 75)
    print(f"{'Filter':<20} {'Mode':<5} {'Dataset':<16} {'Cost':>8}  Status")
    print("-" * 75)
    for fname in sorted(data):
        for mode in ('3d', '2d'):
            for ds, entry in sorted(data.get(fname, {}).get(mode, {}).items()):
                cost = entry.get('cost')
                cost_str = f"{cost:8.3f}" if cost is not None else "       —"
                print(f"{fname:<20} {mode:<5} {ds:<16} {cost_str}  OK")
    print("=" * 75)
    print()
