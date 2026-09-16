---
name: MolNexus
description: Resolve MolNexus compound names or aliases to CIDs, map Gene_symbols to UniProtIDs, query the local BioNexKG in compound-to-entity or protein-to-compound directions, prepare and run independent CPI/CGI/CMI prediction packages, or perform Enrichr pathway enrichment for a compound's observed CGI-U and CGI-D genes with separate SVG bubble plots. Use when a user asks for known molecular relationships, local CPI/CGI/CMI predictions, or pathway enrichment of observed compound-perturbed genes.
---

# BioNexKG Query, MolNexus Prediction, and Pathway Enrichment

Use the local KG before prediction. Keep database evidence, enrichment of observed genes, and model predictions clearly separated.

## Follow the query workflow

1. Identify the query direction and use the matching deterministic entry point:
   - Exact compound CID: run `scripts/query_kg.py --compound-id <CID> --kind <kind>`.
   - Compound `cmap_name` or `compound_aliases`: run `scripts/query_names.py --compound-name <name> --kind <kind>`.
   - Target `Gene_symbol`: run `scripts/query_names.py --gene-symbol <symbol>` to resolve UniProtID and return CPI compounds.
2. Treat matching as exact after Unicode, case, and whitespace normalization. Never fuzzy-match names or split the alias field on punctuation.
3. If compound-name resolution returns `needs_user_input`, show every candidate CID and name, then ask the user to choose. Never select an ambiguous CID automatically.
4. If KG rows exist, answer from those rows and label them as KG results. Do not run prediction unless the user explicitly asks for model prediction.
5. For observed CGI pathway enrichment, resolve an exact CID first, then run `scripts/pathway_enrichment.py`. Do not substitute predicted CGI rows.
6. For prediction, select exactly one task-specific wrapper and read `references/prediction_interfaces.md`.
7. Run prediction wrappers without `--execute` first. Add `--execute` only after the user explicitly requests real inference.
8. Report KG evidence, observed-gene enrichment, and model predictions separately. Never describe any of them as experimental validation.

## Query the KG

```bash
python scripts/query_kg.py --compound-id 119 --kind gene
python scripts/query_names.py --compound-name "L-citrulline" --kind target
python scripts/query_names.py --compound-name "chlorphenesin-carbamate" --kind all
python scripts/query_names.py --gene-symbol FYN --limit 10000
```

Compound-name results include the resolved CID and source field. CPI target results include mapped `gene_symbol`/`gene_symbols` where available. Gene-symbol reverse queries include the resolved UniProtID and matching compound CID, name, alias, and standardized SMILES.

Read `references/kg_query.md` for schemas, exact matching rules, ambiguity handling, and JSON statuses.

## Analyze observed CGI pathways

Run pathway enrichment when the user asks which pathways are associated with a compound's observed up-regulated and down-regulated genes in BioNexKG. Resolve names to one unambiguous CID before invoking this entry point.

```bash
python scripts/pathway_enrichment.py --head 46220502
```

The script extracts observed `CGI-U` and `CGI-D` gene symbols independently, queries the requested Enrichr libraries, writes enrichment CSVs, and creates a separate SVG bubble plot for each relation and library. The x-axis is `-log10(FDR)`, bubble size is Gene Count, and color is Gene Ratio.

Read `references/pathway_enrichment.md` for parameters, output names, dependencies, network behavior, and error handling.

## Prepare predictions

Use separate entry points; do not route task semantics through another task:

```bash
python scripts/predict_cpi.py --help
python scripts/predict_cgi.py --help
python scripts/predict_cmi.py --help
```

Each wrapper validates local inputs and checkpoints, delegates to `python -m <task>.predict`, and returns JSON. The default is a dry run. The wrappers do not import model code, load checkpoints, synthesize features, submit Slurm jobs, or run training.

## Enforce boundaries

- Keep model weights, feature tables, KG rows, and private paths local.
- Note that Enrichr requests send the extracted gene symbols to the remote Enrichr service.
- Never enable random missing-feature fallbacks.
- Never fabricate entity IDs, aliases, Gene_symbols, UniProtIDs, features, or enrichment results.
- Never modify project code, data, model files, checkpoints, environments, or Slurm settings while using this skill.
- Do not run long prediction jobs without an explicit user request.
- Require independent validation for clinical, diagnostic, regulatory, or other critical use.
