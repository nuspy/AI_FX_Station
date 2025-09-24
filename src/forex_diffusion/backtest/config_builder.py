from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple


def numeric_range(start: int, stop: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("step deve essere > 0")
    if start > stop:
        raise ValueError("start deve essere <= stop")
    values: List[int] = []
    current = start
    while current <= stop:
        values.append(current)
        current += step
    if values[-1] != stop:
        values.append(stop)
    return values


def build_param_grid(
    numeric_ranges: Dict[str, Tuple[int, int, int]],
    boolean_choices: Dict[str, Iterable[bool]],
    fixed: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    fixed = fixed or {}
    space: Dict[str, List[Any]] = {}
    for key, triplet in numeric_ranges.items():
        start, stop, step = triplet
        if start == stop:
            space[key] = [start]
        else:
            space[key] = numeric_range(start, stop, step)
    for key, choices in boolean_choices.items():
        vals = list(dict.fromkeys(bool(c) for c in choices))
        if not vals:
            vals = [True, False]
        space[key] = vals
    for key, value in fixed.items():
        space[key] = [value]
    if not space:
        return [dict(fixed)]
    keys = sorted(space.keys())
    grid: List[Dict[str, Any]] = []
    for combination in product(*[space[k] for k in keys]):
        grid.append({k: v for k, v in zip(keys, combination)})
    return grid


def expand_indicator_timeframes(selection: Dict[str, List[str]]) -> List[Dict[str, List[str]]]:
    if not selection:
        return [{}]
    cleaned: Dict[str, List[str]] = {}
    for ind, tfs in selection.items():
        norm = []
        for tf in tfs:
            tf = str(tf).strip()
            if tf and tf not in norm:
                norm.append(tf)
        if norm:
            cleaned[ind] = norm
    if not cleaned:
        return [{}]
    combos: List[Dict[str, List[str]]] = []
    combos.append(cleaned)
    per_indicator = []
    for ind, tfs in cleaned.items():
        per_indicator.append([(ind, [tf]) for tf in tfs])
    for variant in product(*per_indicator):
        combo = {ind: tfs for ind, tfs in variant}
        combos.append(combo)
    unique: Dict[Tuple[Tuple[str, Tuple[str, ...]], ...], Dict[str, List[str]]] = {}
    for combo in combos:
        key = tuple(sorted((k, tuple(v)) for k, v in combo.items()))
        unique[key] = combo
    return list(unique.values())

