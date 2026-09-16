#!/usr/bin/env python3
"""Query compound relations from the local MolNexus KG without prediction."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_KG_FILE = Path(
    os.environ.get(
        "MOLNEXUS_KG_FILE",
        "demo/BioNexKG_demo/BioNexKG_demo.csv",
    )
)
REQUIRED_COLUMNS = {"head", "relation", "tail", "head_type", "tail_type"}
GROUPS = {
    "cpi": {"relations": {"CPI"}, "other_type": "protein"},
    "cgi": {"relations": {"CGI-N", "CGI-D", "CGI-U"}, "other_type": "gene"},
    "cmi": {"relations": {"CMI-N", "CMI-D", "CMI-U"}, "other_type": "metabolite"},
}
KIND_ALIASES = {
    "all": ("cpi", "cgi", "cmi"),
    "target": ("cpi",),
    "protein": ("cpi",),
    "cpi": ("cpi",),
    "gene": ("cgi",),
    "cgi": ("cgi",),
    "metabolite": ("cmi",),
    "cmi": ("cmi",),
}
INTEGER_FLOAT_RE = re.compile(r"^([+-]?\d+)\.0+$")


def canonicalize_compound_id(value: str) -> tuple[str, bool]:
    """Preserve string IDs except for a terminal all-zero decimal suffix."""
    stripped = str(value).strip()
    match = INTEGER_FLOAT_RE.fullmatch(stripped)
    if match:
        return match.group(1), True
    return stripped, stripped != str(value)


def json_print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def error(message: str, *, code: str, details: dict[str, Any] | None = None) -> int:
    json_print(
        {
            "status": "error",
            "error": {"code": code, "message": message, "details": details or {}},
        }
    )
    return 2


def selected_groups(kind: str) -> tuple[str, ...]:
    return KIND_ALIASES[kind]


def query(
    kg_file: Path,
    compound_ids: list[str],
    *,
    kind: str,
    limit: int,
) -> dict[str, Any]:
    normalized: list[str] = []
    normalization: dict[str, dict[str, Any]] = {}
    for requested in compound_ids:
        canonical, changed = canonicalize_compound_id(requested)
        if not canonical:
            raise ValueError("compound ID must not be empty")
        if canonical not in normalized:
            normalized.append(canonical)
        normalization[requested] = {"canonical": canonical, "changed": changed}

    wanted = set(normalized)
    groups = selected_groups(kind)
    relation_to_group = {
        relation: group
        for group in groups
        for relation in GROUPS[group]["relations"]
    }
    present = {compound_id: False for compound_id in normalized}
    matches = {
        compound_id: {group: [] for group in groups}
        for compound_id in normalized
    }
    counts = {
        compound_id: {group: 0 for group in groups}
        for compound_id in normalized
    }
    seen = {compound_id: {group: set() for group in groups} for compound_id in normalized}
    duplicates_ignored = 0
    scanned_rows = 0

    with kg_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise KeyError(f"KG is missing required columns: {missing}")

        for row in reader:
            scanned_rows += 1
            relation = str(row.get("relation", "")).strip()
            head = str(row.get("head", "")).strip()
            tail = str(row.get("tail", "")).strip()
            head_type = str(row.get("head_type", "")).strip()
            tail_type = str(row.get("tail_type", "")).strip()

            endpoints: list[tuple[str, str, str, str]] = []
            if head_type == "compound":
                compound_id, _ = canonicalize_compound_id(head)
                if compound_id in wanted:
                    endpoints.append((compound_id, head, tail, tail_type))
            if tail_type == "compound":
                compound_id, _ = canonicalize_compound_id(tail)
                if compound_id in wanted:
                    endpoints.append((compound_id, tail, head, head_type))

            for compound_id, raw_compound_id, other_id, other_type in endpoints:
                present[compound_id] = True
                group = relation_to_group.get(relation)
                if not group or other_type != GROUPS[group]["other_type"]:
                    continue
                key = (relation, other_id, other_type)
                if key in seen[compound_id][group]:
                    duplicates_ignored += 1
                    continue
                seen[compound_id][group].add(key)
                counts[compound_id][group] += 1
                if len(matches[compound_id][group]) < limit:
                    matches[compound_id][group].append(
                        {
                            "compound_id": raw_compound_id,
                            "relation": relation,
                            "entity_id": other_id,
                            "entity_type": other_type,
                        }
                    )

    result_items = []
    any_present = False
    any_matches = False
    for compound_id in normalized:
        any_present = any_present or present[compound_id]
        total = sum(counts[compound_id].values())
        any_matches = any_matches or total > 0
        result_items.append(
            {
                "compound_id": compound_id,
                "present_in_kg": present[compound_id],
                "counts": counts[compound_id],
                "matches": matches[compound_id],
                "truncated": {
                    group: counts[compound_id][group] > len(matches[compound_id][group])
                    for group in groups
                },
            }
        )

    if any_matches:
        status = "ok"
    elif any_present:
        status = "found_no_requested_relations"
    else:
        status = "not_found"

    return {
        "status": status,
        "source": "knowledge_graph",
        "kg_file": str(kg_file),
        "kind": kind,
        "relation_groups": list(groups),
        "requested_ids": compound_ids,
        "id_normalization": normalization,
        "results": result_items,
        "scanned_rows": scanned_rows,
        "duplicates_ignored": duplicates_ignored,
        "warnings": [
            "These are existing KG relations, not model predictions.",
            "Interpret relation provenance and biological validity separately.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query existing compound CPI/CGI/CMI relations from the local MolNexus KG."
    )
    parser.add_argument(
        "--compound-id",
        action="append",
        required=True,
        help="Exact compound ID. Repeat to query more than one ID.",
    )
    parser.add_argument(
        "--kind",
        choices=sorted(KIND_ALIASES),
        default="all",
        help="Requested relation kind. Default: all.",
    )
    parser.add_argument(
        "--kg-file",
        default=str(DEFAULT_KG_FILE),
        help="KG CSV with head, relation, tail, head_type, and tail_type columns.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum returned unique rows per compound and relation group. Counts remain complete.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kg_file = Path(args.kg_file).expanduser().resolve()
    if not kg_file.is_file():
        return error(
            "KG file does not exist.",
            code="missing_kg_file",
            details={"kg_file": str(kg_file)},
        )
    if args.limit < 1:
        return error("--limit must be at least 1.", code="invalid_limit")
    try:
        payload = query(
            kg_file,
            args.compound_id,
            kind=args.kind,
            limit=args.limit,
        )
    except (OSError, csv.Error, KeyError, ValueError) as exc:
        return error(str(exc), code="kg_query_failed")
    json_print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
