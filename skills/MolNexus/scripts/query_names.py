#!/usr/bin/env python3
"""Resolve compound names or gene symbols and query existing KG CPI/CGI/CMI rows."""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from pathlib import Path
from typing import Any

from query_kg import (
    DEFAULT_KG_FILE,
    KIND_ALIASES,
    canonicalize_compound_id,
    query,
    selected_groups,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_COMPOUND_REGISTRY = (
    PROJECT_ROOT / "query_compounds_with_aliases.csv"
)
DEFAULT_PROTEIN_MAPPING = PROJECT_ROOT / "protein_genes_idmapping.tsv"
COMPOUND_COLUMNS = {"CID", "cmap_name", "compound_aliases", "standardized_smiles"}
PROTEIN_COLUMNS = {"UniprotID", "Gene_symbol"}
KG_COLUMNS = {"head", "relation", "tail", "head_type", "tail_type"}


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def fail(message: str, *, code: str, details: dict[str, Any] | None = None) -> int:
    emit(
        {
            "status": "error",
            "error": {"code": code, "message": message, "details": details or {}},
        }
    )
    return 2


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value)).strip()
    return " ".join(normalized.split()).casefold()


def require_columns(path: Path, actual: list[str] | None, required: set[str]) -> None:
    missing = sorted(required - set(actual or []))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")


def load_compounds(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, set[str]]]]:
    by_cid: dict[str, dict[str, str]] = {}
    name_lookup: dict[str, dict[str, set[str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(path, reader.fieldnames, COMPOUND_COLUMNS)
        for row in reader:
            cid, _ = canonicalize_compound_id(row.get("CID", ""))
            if not cid:
                continue
            record = {
                "CID": cid,
                "cmap_name": str(row.get("cmap_name", "")).strip(),
                "compound_aliases": str(row.get("compound_aliases", "")).strip(),
                "standardized_smiles": str(row.get("standardized_smiles", "")).strip(),
            }
            by_cid.setdefault(cid, record)
            values_by_field = {
                "cmap_name": [record["cmap_name"]],
                "compound_aliases": record["compound_aliases"].split("|"),
            }
            for field, values in values_by_field.items():
                for value in values:
                    value = value.strip()
                    if not value:
                        continue
                    key = normalize_text(value)
                    name_lookup.setdefault(key, {}).setdefault(cid, set()).add(field)
    return by_cid, name_lookup


def resolve_compound(
    requested: str,
    by_cid: dict[str, dict[str, str]],
    lookup: dict[str, dict[str, set[str]]],
) -> dict[str, Any]:
    candidates = []
    for cid, fields in lookup.get(normalize_text(requested), {}).items():
        candidate = dict(by_cid[cid])
        candidate["matched_fields"] = sorted(fields)
        candidates.append(candidate)
    candidates.sort(key=lambda item: item["CID"])
    status = "resolved" if len(candidates) == 1 else "ambiguous" if candidates else "not_found"
    return {"query": requested, "status": status, "candidates": candidates}


def load_proteins(
    path: Path,
) -> tuple[dict[str, list[str]], dict[str, list[dict[str, str]]]]:
    symbols_by_uniprot: dict[str, list[str]] = {}
    symbol_lookup: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_columns(path, reader.fieldnames, PROTEIN_COLUMNS)
        for row in reader:
            uniprot_id = str(row.get("UniprotID", "")).strip()
            gene_symbol = str(row.get("Gene_symbol", "")).strip()
            if not uniprot_id or not gene_symbol:
                continue
            symbols = symbols_by_uniprot.setdefault(uniprot_id, [])
            if gene_symbol not in symbols:
                symbols.append(gene_symbol)
            records = symbol_lookup.setdefault(normalize_text(gene_symbol), [])
            record = {"UniprotID": uniprot_id, "Gene_symbol": gene_symbol}
            if record not in records:
                records.append(record)
    return symbols_by_uniprot, symbol_lookup


def enrich_compound_query(
    payload: dict[str, Any],
    compounds: dict[str, dict[str, str]],
    symbols_by_uniprot: dict[str, list[str]],
) -> None:
    for result in payload.get("results", []):
        metadata = compounds.get(result["compound_id"])
        if metadata:
            result["compound_metadata"] = metadata
        for matches in result.get("matches", {}).values():
            for match in matches:
                if match.get("entity_type") != "protein":
                    continue
                symbols = list(symbols_by_uniprot.get(match["entity_id"], []))
                match["gene_symbols"] = symbols
                if len(symbols) == 1:
                    match["gene_symbol"] = symbols[0]


def query_compound_name(
    *,
    requested: str,
    kg_file: Path,
    compound_file: Path,
    protein_file: Path,
    kind: str,
    limit: int,
) -> dict[str, Any]:
    compounds, lookup = load_compounds(compound_file)
    resolution = resolve_compound(requested, compounds, lookup)
    if resolution["status"] != "resolved":
        return {
            "status": "needs_user_input" if resolution["status"] == "ambiguous" else "not_found",
            "source": "compound_registry",
            "query_direction": "compound_name_to_entity",
            "compound_resolution": resolution,
            "results": [],
            "scanned_rows": 0,
        }

    cid = resolution["candidates"][0]["CID"]
    symbols_by_uniprot: dict[str, list[str]] = {}
    if "cpi" in selected_groups(kind):
        symbols_by_uniprot, _ = load_proteins(protein_file)
    payload = query(kg_file, [cid], kind=kind, limit=limit)
    payload["query_direction"] = "compound_name_to_entity"
    payload["compound_resolution"] = resolution
    enrich_compound_query(payload, compounds, symbols_by_uniprot)
    return payload


def query_gene_symbol(
    *,
    requested: str,
    kg_file: Path,
    compound_file: Path,
    protein_file: Path,
    limit: int,
) -> dict[str, Any]:
    compounds, _ = load_compounds(compound_file)
    _, symbol_lookup = load_proteins(protein_file)
    mappings = list(symbol_lookup.get(normalize_text(requested), []))
    mappings.sort(key=lambda item: item["UniprotID"])
    resolution = {
        "query": requested,
        "status": "resolved" if mappings else "not_found",
        "mappings": mappings,
    }
    if not mappings:
        return {
            "status": "not_found",
            "source": "protein_mapping",
            "query_direction": "gene_symbol_to_compound",
            "gene_resolution": resolution,
            "results": [],
            "scanned_rows": 0,
        }

    wanted = {item["UniprotID"] for item in mappings}
    present = False
    count = 0
    matches = []
    seen = set()
    scanned_rows = 0
    with kg_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require_columns(kg_file, reader.fieldnames, KG_COLUMNS)
        for row in reader:
            scanned_rows += 1
            relation = str(row.get("relation", "")).strip()
            head = str(row.get("head", "")).strip()
            tail = str(row.get("tail", "")).strip()
            head_type = str(row.get("head_type", "")).strip()
            tail_type = str(row.get("tail_type", "")).strip()

            endpoints = []
            if head_type == "protein" and head in wanted:
                endpoints.append((head, tail, tail_type))
            if tail_type == "protein" and tail in wanted:
                endpoints.append((tail, head, head_type))

            for uniprot_id, other_id, other_type in endpoints:
                present = True
                if relation != "CPI" or other_type != "compound":
                    continue
                compound_id, _ = canonicalize_compound_id(other_id)
                key = (uniprot_id, compound_id)
                if key in seen:
                    continue
                seen.add(key)
                count += 1
                if len(matches) < limit:
                    metadata = compounds.get(compound_id, {})
                    matches.append(
                        {
                            "gene_symbol_query": requested,
                            "uniprot_id": uniprot_id,
                            "relation": "CPI",
                            "compound_id": compound_id,
                            "cmap_name": metadata.get("cmap_name"),
                            "compound_aliases": metadata.get("compound_aliases"),
                            "standardized_smiles": metadata.get("standardized_smiles"),
                        }
                    )

    status = "ok" if count else "found_no_requested_relations" if present else "not_found"
    return {
        "status": status,
        "source": "knowledge_graph",
        "query_direction": "gene_symbol_to_compound",
        "kg_file": str(kg_file),
        "gene_resolution": resolution,
        "results": [
            {
                "gene_symbol_query": requested,
                "resolved_uniprot_ids": [item["UniprotID"] for item in mappings],
                "protein_present_in_kg": present,
                "count": count,
                "matches": matches,
                "truncated": count > len(matches),
            }
        ],
        "scanned_rows": scanned_rows,
        "warnings": [
            "Gene symbols are matched exactly after case and whitespace normalization.",
            "These are existing KG CPI relations, not model predictions.",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resolve a compound name/alias or Gene_symbol and query the MolNexus KG."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--compound-name", help="Exact cmap_name or compound_aliases value.")
    group.add_argument("--gene-symbol", help="Exact Gene_symbol for reverse CPI lookup.")
    parser.add_argument(
        "--kind",
        choices=sorted(KIND_ALIASES),
        default="all",
        help="Relation kind for --compound-name. Ignored for --gene-symbol.",
    )
    parser.add_argument("--kg-file", default=str(DEFAULT_KG_FILE))
    parser.add_argument("--compound-registry", default=str(DEFAULT_COMPOUND_REGISTRY))
    parser.add_argument("--protein-mapping", default=str(DEFAULT_PROTEIN_MAPPING))
    parser.add_argument("--limit", type=int, default=10000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        return fail("--limit must be at least 1.", code="invalid_limit")

    kg_file = Path(args.kg_file).expanduser().resolve()
    compound_file = Path(args.compound_registry).expanduser().resolve()
    protein_file = Path(args.protein_mapping).expanduser().resolve()
    for path, code in (
        (kg_file, "missing_kg_file"),
        (compound_file, "missing_compound_registry"),
        (protein_file, "missing_protein_mapping"),
    ):
        if not path.is_file():
            return fail("Required local file does not exist.", code=code, details={"path": str(path)})

    try:
        if args.compound_name is not None:
            payload = query_compound_name(
                requested=args.compound_name,
                kg_file=kg_file,
                compound_file=compound_file,
                protein_file=protein_file,
                kind=args.kind,
                limit=args.limit,
            )
        else:
            payload = query_gene_symbol(
                requested=args.gene_symbol,
                kg_file=kg_file,
                compound_file=compound_file,
                protein_file=protein_file,
                limit=args.limit,
            )
    except (OSError, csv.Error, KeyError, ValueError) as exc:
        return fail(str(exc), code="name_query_failed")

    emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
