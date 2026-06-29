"""Utilities for settings patch/load workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypeAlias, TypeGuard, cast

from services.services_utils import JSONValue

JSONObject: TypeAlias = dict[str, JSONValue]


def _is_json_value(value: object) -> TypeGuard[JSONValue]:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        typed_items = cast(list[object], value)
        return all(_is_json_value(item) for item in typed_items)
    if isinstance(value, dict):
        typed_mapping = cast(dict[object, object], value)
        return all(isinstance(key, str) and _is_json_value(item) for key, item in typed_mapping.items())
    return False


def _is_json_object(value: object) -> TypeGuard[JSONObject]:
    if not isinstance(value, dict):
        return False
    typed_mapping = cast(dict[object, object], value)
    return all(isinstance(key, str) and _is_json_value(item) for key, item in typed_mapping.items())


def ensure_json_object(payload: object) -> JSONObject:
    if not _is_json_object(payload):
        raise ValueError("Settings payload must be a JSON object")
    return payload


def deep_merge_dicts(base: Mapping[str, JSONValue], patch: Mapping[str, JSONValue]) -> JSONObject:
    merged: JSONObject = dict(base)
    for key, value in patch.items():
        base_value = merged.get(key)
        if _is_json_object(value) and _is_json_object(base_value):
            merged[key] = deep_merge_dicts(base_value, value)
        else:
            merged[key] = value
    return merged


def strip_none_values(payload: Mapping[str, JSONValue]) -> JSONObject:
    cleaned: JSONObject = {}
    for key, value in payload.items():
        if value is None:
            continue
        if _is_json_object(value):
            cleaned[key] = strip_none_values(value)
        else:
            cleaned[key] = value
    return cleaned


def collect_changed_paths(before: JSONValue, after: JSONValue, prefix: str = "") -> set[str]:
    if _is_json_object(before) and _is_json_object(after):
        paths: set[str] = set()
        for key in set(before.keys()) | set(after.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else key
            if key not in before or key not in after:
                paths.add(next_prefix)
                continue
            paths |= collect_changed_paths(before[key], after[key], next_prefix)
        return paths

    if before != after and prefix:
        return {prefix}
    return set()


def migrate_legacy_settings(raw: Mapping[str, JSONValue]) -> JSONObject:
    migrated: JSONObject = dict(raw)
    if (
        "prompt_enhancer_enabled" in migrated
        and "prompt_enhancer_enabled_t2v" not in migrated
    ):
        legacy_value = bool(migrated["prompt_enhancer_enabled"])
        migrated.setdefault("prompt_enhancer_enabled_t2v", legacy_value)
        migrated.setdefault("prompt_enhancer_enabled_i2v", legacy_value)

    migrated.pop("prompt_enhancer_enabled", None)
    return migrated
