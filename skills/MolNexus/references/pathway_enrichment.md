# Pathway enrichment interface

## Scope

Use `scripts/pathway_enrichment.py` to analyze observed compound-gene interactions in BioNexKG. For one compound/head ID, the script extracts `CGI-U` and `CGI-D` genes and analyzes the two lists independently. Do not describe these results as enrichment of model-predicted CGI.

A KG row is eligible only when `head` matches the requested ID, `head_type` is `compound`, `tail_type` is `gene`, and `relation` is `CGI-U` or `CGI-D`.

## Entry point

From the MolNexus skill directory:

```bash
python scripts/pathway_enrichment.py --head 46220502
```

To select all main inputs explicitly:

```bash
python scripts/pathway_enrichment.py \
  --head 46220502 \
  --kg /path/to/BioNexKG.csv \
  --outdir results/pathway_enrichment/46220502 \
  --organism human \
  --libraries MSigDB_Hallmark_2020 \
  --top_terms 5 \
  --cmap plasma
```

`--compound-id`, `--kg-file`, and `--output-directory` are aliases for `--head`, `--kg`, and `--outdir`. Run `python scripts/pathway_enrichment.py --help` for plot-size and chunking options.

## Main parameters

- `--head`: exact compound/head ID. The bundled demo default is `46220502`.
- `--kg`: KG CSV path. The default is the project's `demo/BioNexKG_demo/BioNexKG_demo.csv`.
- `--outdir`: output directory; missing parent directories are created.
- `--organism`: organism accepted by gseapy/Enrichr; default `human`.
- `--libraries`: one or more exact Enrichr library names; default `MSigDB_Hallmark_2020`.
- `--top_terms`: maximum terms shown in each plot; default `10`.
- `--cmap`: Matplotlib colormap for Gene Ratio; default `plasma`.

## Outputs

For compound `<CID>`, the output directory contains:

- `head_<CID>_observed_CGI_U_D_rows.csv`: deduplicated observed KG rows.
- `head_<CID>_observed_CGI-U_genes.txt` and `head_<CID>_observed_CGI-D_genes.txt`: unique uppercase gene lists, including empty files when one relation has no genes.
- `head_<CID>_observed_<RELATION>_<LIBRARY>_enrichr.csv`: complete standardized Enrichr results for each non-empty gene list. A valid no-hit response produces an empty CSV with stable headers.
- `head_<CID>_observed_<RELATION>_<LIBRARY>_single_dotplot.svg`: independent CGI-U or CGI-D plot when plottable terms exist.

Plots preserve the tested implementation: x is `-log10(FDR)`, bubble area represents Gene Count, color represents Gene Ratio, and output is SVG.

## Dependencies and network

The runtime needs `numpy`, `pandas`, `matplotlib`, and `gseapy`. This project has no requirements, pyproject, or environment manifest to update, so install missing packages in the active environment, for example:

```bash
python -m pip install numpy pandas matplotlib gseapy
```

Enrichr library discovery and enrichment require network access to the Enrichr service. A missing package, unavailable library, unsupported organism, or failed network request returns a concise error and a nonzero exit status. The script contains no Slurm, conda, proxy, or email configuration.

A compound absent as a compound head is an error. If the compound exists but one relation has no genes, the other relation still runs. If neither relation has genes, the observed-row and both gene-list files are written and enrichment is skipped successfully.
