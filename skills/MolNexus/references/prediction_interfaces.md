# Prediction interfaces

All wrappers resolve relative paths against the MolNexus project root. Set `--project-root` or `MOLNEXUS_PROJECT_ROOT` when the skill is copied elsewhere. They return JSON and default to validation/dry-run; `--execute` starts real inference.

## CPI

Use `scripts/predict_cpi.py` for compound-protein interaction target prediction or compound screening.

Required:

- `--split`: `Warm`, `CCS`, `DCS`, or `PCS`; selects both model hyperparameters and `model/cpi/<split>/ensemble_N.pt`.
- `--compounds`: prepared compound list or triplet table.
- `--feat`: real external compound features.
- `--out-dir`: result directory.
- `--candidate-file`: protein candidate table; this is the only candidate source.

Example:

```bash
python scripts/predict_cpi.py \
  --split CCS \
  --task-mode target_prediction \
  --compounds /path/to/compounds.csv \
  --feat /path/to/compound_kpgt.csv \
  --candidate-file /path/to/proteins.csv \
  --out-dir results/cpi/<run_name>
```

Expected outputs are `ensemble_prob_matrix.csv`, `ensemble_rank_matrix.csv`, `ensemble_label_matrix.csv`, and `ensemble_label_vote5_matrix.csv`.

## CGI

Use `scripts/predict_cgi.py` for `CGI-N`, `CGI-D`, and `CGI-U` prediction.

Required:

- `--input-csv`: prepared compound-gene triplets or feature table.
- `--compound-features`: real compound KPGT feature table passed to the package's `--cpd_features_file`.
- `--output-dir`: result directory.

The default model directory is `model/cgi`. For triplets, normally set `--compound-id-col` and `--gene-id-col` explicitly. The wrapper deliberately has no random-feature fallback option.

```bash
python scripts/predict_cgi.py \
  --input-csv /path/to/compound_gene.csv \
  --compound-features /path/to/compound_kpgt.csv \
  --input-format triplet \
  --compound-id-col head \
  --gene-id-col tail \
  --output-dir results/cgi/<run_name> \
  --device cpu
```

The main output is `predictions.csv`. Evaluation files are created only when labels are available.

## CMI

Use `scripts/predict_cmi.py` for `CMI-N`, `CMI-D`, and `CMI-U` prediction.

Required:

- `--input-csv`: prepared compound-metabolite triplets or feature table.
- `--output-dir`: result directory.

The default model directory is `model/cmi`. Supply `--external-compound-features` for cold/external compounds and `--external-metabolite-features` for external metabolites. Omission is valid only if every requested entity already has a real feature in the configured base tables. The task package raises on missing features; the wrapper never enables random fallback.

```bash
python scripts/predict_cmi.py \
  --input-csv /path/to/compound_metabolite.csv \
  --external-compound-features /path/to/compound_kpgt.csv \
  --input-format triplet \
  --compound-id-col head \
  --metabolite-id-col tail \
  --output-dir results/cmi/<run_name> \
  --device cpu
```

The main output is `predictions.csv`. Evaluation files are created only when labels are available.

## Execution policy

Inspect a successful `dry_run` response before adding `--execute`. Do not add random-feature flags, do not load checkpoints for validation, and do not submit Slurm jobs through these wrappers. If advanced background graph or feature overrides are required, inspect the relevant task's `python -m <task>.predict --help` and pass only authoritative local files.
