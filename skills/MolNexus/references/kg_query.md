# KG query contract

## Local mapping sources

Use only these project-local mapping files:

- `query_compounds_with_aliases.csv`
  - columns: `CID,cmap_name,compound_aliases,standardized_smiles`
  - resolves an exact compound name or alias to CID
- `protein_genes_idmapping.tsv`
  - columns: `UniprotID,Gene_symbol`
  - enriches CPI protein results and resolves Gene_symbol for reverse CPI lookup

The KG itself must contain:

```text
head,relation,tail,head_type,tail_type
```

## Entry points

Query an exact CID:

```bash
python scripts/query_kg.py --compound-id 833 --kind target
```

Resolve a `cmap_name` or `compound_aliases` value, then query:

```bash
python scripts/query_names.py --compound-name "L-citrulline" --kind target
python scripts/query_names.py --compound-name "chlorphenesin-carbamate" --kind all
```

Resolve a target Gene_symbol to UniProtID, then return CPI compounds:

```bash
python scripts/query_names.py --gene-symbol FYN --limit 10000
```

## Relation mapping

| Request | `--kind` | KG relation | Other entity type |
|---|---|---|---|
| compound targets | `target` or `cpi` | `CPI` | `protein` |
| perturbed genes | `gene` or `cgi` | `CGI-N`, `CGI-D`, `CGI-U` | `gene` |
| perturbed metabolites | `metabolite` or `cmi` | `CMI-N`, `CMI-D`, `CMI-U` | `metabolite` |

## Exact matching rules

- Normalize Unicode with NFKC, strip surrounding whitespace, collapse internal whitespace, and compare case-insensitively.
- Match `cmap_name` and each non-empty, `|`-delimited `compound_aliases` value independently.
- Do not split `compound_aliases` on commas or punctuation other than `|` because punctuation may be part of an alias.
- Do not perform fuzzy, substring, SMILES-similarity, or external name matching.
- If one name maps to multiple CIDs, return `needs_user_input` with every candidate and do not query the KG until the user chooses.
- Match Gene_symbol exactly after the same case and whitespace normalization.
- Preserve all UniProtID mappings for a Gene_symbol; do not silently overwrite one-to-many mappings.
- Preserve compound IDs as strings. Canonicalize only an all-zero decimal suffix such as `100018.0` to `100018`.

## Returned fields

Compound-name queries return:

- `compound_resolution.status`: `resolved`, `ambiguous`, or `not_found`
- candidate `CID`, `cmap_name`, `compound_aliases`, and `standardized_smiles`
- `matched_fields` showing whether the exact match came from name or alias
- normal KG relation counts and matches
- CPI matches enriched with `gene_symbol` or `gene_symbols`

Gene-symbol queries return:

- exact `Gene_symbol` to `UniprotID` mappings
- matching CPI compound IDs
- compound name, alias, and standardized SMILES when present in the compound mapping file
- complete unique count and a `truncated` flag

## JSON status

- `ok`: at least one requested KG relation was found.
- `needs_user_input`: a compound name or alias maps to multiple CIDs.
- `found_no_requested_relations`: the resolved entity is in the KG but has no requested relation.
- `not_found`: the name/symbol was not mapped or the resolved entity was absent from the KG.
- `error`: a required file, schema, or query was invalid.

Counts cover all unique matches. Each query returns at most `--limit` rows while retaining complete counts.
