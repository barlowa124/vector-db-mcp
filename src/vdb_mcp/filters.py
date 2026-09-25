"""Pinecone-style metadata filtering.

Top-level dict is an implicit AND. Field values may be a scalar (implicit
$eq) or an operator dict. Supported operators: $eq $ne $gt $gte $lt $lte
$in $nin $exists $and $or.
"""

_OPS = ("$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin",
        "$exists")


def _cmp(value, op, operand):
    if op == "$exists":
        return (value is not None) == bool(operand)
    if value is None:
        return False
    if op == "$eq":
        return value == operand
    if op == "$ne":
        return value != operand
    if op in ("$in", "$nin"):
        try:
            inside = value in operand
        except TypeError:
            return False  # non-iterable operand: never match
        return inside if op == "$in" else not inside
    try:
        if op == "$gt":
            return value > operand
        if op == "$gte":
            return value >= operand
        if op == "$lt":
            return value < operand
        if op == "$lte":
            return value <= operand
    except TypeError:
        return False
    raise ValueError(f"unknown operator {op}")


def match(metadata: dict, flt: dict | None) -> bool:
    if not flt:
        return True
    metadata = metadata or {}
    for key, cond in flt.items():
        if key == "$and":
            if not all(match(metadata, c) for c in cond):
                return False
            continue
        if key == "$or":
            if not any(match(metadata, c) for c in cond):
                return False
            continue
        value = metadata.get(key)
        if isinstance(cond, dict):
            for op, operand in cond.items():
                if op not in _OPS:
                    raise ValueError(f"unknown operator {op}")
                if not _cmp(value, op, operand):
                    return False
        elif value != cond:
            return False
    return True
