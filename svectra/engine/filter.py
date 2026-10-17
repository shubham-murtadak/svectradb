"""
Metadata filtering engine for SvectraDB.
Supports MongoDB-style declarative filter expressions.
"""

from typing import Any, Callable, Dict, List, Optional


def _get_nested_value(data: Dict[str, Any], path: str) -> Any:
    """Retrieve value from a nested dictionary using dot notation (e.g. 'author.name')."""
    if not isinstance(data, dict):
        return None
    parts = path.split(".")
    curr = data
    for part in parts:
        if isinstance(curr, dict) and part in curr:
            curr = curr[part]
        else:
            return None
    return curr


def match_filter(metadata: Optional[Dict[str, Any]], filter_expr: Optional[Dict[str, Any]]) -> bool:
    """
    Evaluate if a metadata dictionary satisfies the given filter expression.
    If filter_expr is empty or None, returns True.
    """
    if not filter_expr:
        return True
    if metadata is None:
        metadata = {}

    for key, condition in filter_expr.items():
        if key == "$and":
            if not isinstance(condition, list):
                raise ValueError("$and must be a list of filter conditions")
            if not all(match_filter(metadata, sub) for sub in condition):
                return False

        elif key == "$or":
            if not isinstance(condition, list):
                raise ValueError("$or must be a list of filter conditions")
            if not any(match_filter(metadata, sub) for sub in condition):
                return False

        elif key == "$not":
            if not isinstance(condition, dict):
                raise ValueError("$not must be a filter condition dict")
            if match_filter(metadata, condition):
                return False

        else:
            # Field comparison
            val = _get_nested_value(metadata, key)
            if not _evaluate_condition(val, condition):
                return False

    return True


def _evaluate_condition(val: Any, condition: Any) -> bool:
    """Evaluate a single field condition."""
    # Direct equality check if condition is not an operator dict
    if not isinstance(condition, dict):
        # If val is a list, check if condition is in list
        if isinstance(val, (list, tuple, set)):
            return condition in val
        return val == condition

    for op, target in condition.items():
        if op == "$eq":
            if isinstance(val, (list, tuple, set)):
                if target not in val:
                    return False
            elif val != target:
                return False

        elif op == "$ne":
            if isinstance(val, (list, tuple, set)):
                if target in val:
                    return False
            elif val == target:
                return False

        elif op == "$gt":
            if val is None or not (val > target):
                return False

        elif op == "$gte":
            if val is None or not (val >= target):
                return False

        elif op == "$lt":
            if val is None or not (val < target):
                return False

        elif op == "$lte":
            if val is None or not (val <= target):
                return False

        elif op == "$in":
            if not isinstance(target, (list, tuple, set)):
                raise ValueError("$in target must be a list/set")
            if isinstance(val, (list, tuple, set)):
                if not any(item in target for item in val):
                    return False
            elif val not in target:
                return False

        elif op == "$nin":
            if not isinstance(target, (list, tuple, set)):
                raise ValueError("$nin target must be a list/set")
            if isinstance(val, (list, tuple, set)):
                if any(item in target for item in val):
                    return False
            elif val in target:
                return False

        elif op == "$contains":
            if val is None or not isinstance(val, (str, list, tuple)):
                return False
            if str(target) not in val:
                return False

        elif op == "$startswith":
            if val is None or not isinstance(val, str):
                return False
            if not val.startswith(str(target)):
                return False

        elif op == "$endswith":
            if val is None or not isinstance(val, str):
                return False
            if not val.endswith(str(target)):
                return False

        elif op == "$exists":
            exists = val is not None
            if exists != bool(target):
                return False

        else:
            raise ValueError(f"Unknown filter operator: {op}")

    return True


def compile_filter(filter_expr: Optional[Dict[str, Any]]) -> Callable[[Optional[Dict[str, Any]]], bool]:
    """Compile a filter expression into a fast reusable callable predicate."""
    if not filter_expr:
        return lambda meta: True
    return lambda meta: match_filter(meta, filter_expr)
