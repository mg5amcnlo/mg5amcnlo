# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import itertools
import re
from collections.abc import Mapping, Sequence

ParticleName = str
ProcessTuple = tuple[ParticleName, ...]
OrderTuple = tuple[int, ...]


def split_process_set(process_string: str) -> tuple[str, ...]:
    """Split process sets without treating bars inside brackets as separators."""

    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(process_string):
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise ValueError("unmatched ']' in process string")
        elif char == "|" and depth == 0:
            part = process_string[start:index].strip()
            if not part:
                raise ValueError("empty process in process set")
            parts.append(part)
            start = index + 1
    if depth != 0:
        raise ValueError("unmatched '[' in process string")
    tail = process_string[start:].strip()
    if not tail:
        raise ValueError("empty process in process set")
    parts.append(tail)
    return tuple(parts)


def expand_process_variants(
    process_string: str,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Expand anonymous multiparticle slots and repetition syntax.

    Built-in inclusive labels such as ``p`` and ``j`` are kept symbolic for the
    enumerator. Anonymous slots like ``[d g]`` are expanded by cartesian product,
    and each repeated slot in ``3*[d g]`` is treated independently.
    Optional named aliases are expanded through the same token-aware path.
    """

    variants: list[str] = []
    for process in split_process_set(process_string):
        parts = process.lower().replace("bar", "~").split(">")
        if len(parts) != 2:
            raise ValueError("invalid collision format; expected 'initial > final'")
        initial_options = _expand_side_tokens(
            _tokenize_side(parts[0].strip()), aliases=aliases
        )
        final_options = _expand_side_tokens(
            _tokenize_side(parts[1].strip()), aliases=aliases
        )
        for initial in itertools.product(*initial_options):
            for final in itertools.product(*final_options):
                variants.append(f"{' '.join(initial)} > {' '.join(final)}")
    return tuple(dict.fromkeys(variants))


def canonical_process_key(process: str) -> str:
    tokens = process.lower().replace("bar", "~").replace(">", " > ").split()
    safe = []
    for token in tokens:
        if token == ">":
            safe.append("to")
        else:
            safe.append(
                token.replace("~", "bar").replace("+", "plus").replace("-", "minus")
            )
    return "_".join(safe)


def _tokenize_side(side: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(side):
        if side[index].isspace():
            index += 1
            continue
        if side[index] == "[":
            end = side.find("]", index + 1)
            if end < 0:
                raise ValueError("unmatched '[' in process string")
            tokens.append(side[index : end + 1])
            index = end + 1
            continue
        end = index
        while end < len(side) and not side[end].isspace():
            if side[end] == "[":
                bracket = side.find("]", end + 1)
                if bracket < 0:
                    raise ValueError("unmatched '[' in process string")
                end = bracket + 1
            else:
                end += 1
        tokens.append(side[index:end])
        index = end
    return tokens


def _expand_side_tokens(
    tokens: Sequence[str],
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[tuple[str, ...], ...]:
    expanded: list[tuple[str, ...]] = []
    normalized_aliases = {
        str(name).casefold(): tuple(str(value).casefold() for value in values)
        for name, values in (aliases or {}).items()
    }
    for token in tokens:
        repeat, item = _split_repeat_token(token)
        options = tuple(
            member
            for option in _anonymous_options(item)
            for member in normalized_aliases.get(option.casefold(), (option,))
        )
        expanded.extend(options for _ in range(repeat))
    return tuple(expanded)


def _split_repeat_token(token: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)\*(.+)", token)
    if match:
        return int(match.group(1)), match.group(2)
    compact = re.fullmatch(r"(\d+)([A-Za-z][A-Za-z0-9+~\-]*)", token)
    if compact:
        return int(compact.group(1)), compact.group(2)
    return 1, token


def _anonymous_options(token: str) -> tuple[str, ...]:
    if token.startswith("[") and token.endswith("]"):
        options = tuple(_tokenize_side(token[1:-1].strip()))
        if not options:
            raise ValueError("anonymous multiparticle label cannot be empty")
        return options
    return (token,)


__all__ = [
    "OrderTuple",
    "ParticleName",
    "ProcessTuple",
    "canonical_process_key",
    "expand_process_variants",
    "split_process_set",
]
