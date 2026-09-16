from __future__ import annotations

import argparse
from .config import DEFAULTS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "External compound-metabolite three-class prediction with the "
            "5-fold HGT CMI ensemble."
        )
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default=DEFAULTS["output_dir"])
    parser.add_argument("--model_dir", default=DEFAULTS["model_dir"])
    parser.add_argument(
        "--input_format",
        choices=["triplet", "kpgt_features", "auto"],
        default="auto",
    )
    parser.add_argument("--label_col")
    parser.add_argument("--compound_id_col")
    parser.add_argument("--metabolite_id_col")
    parser.add_argument("--compound_name_col")
    parser.add_argument("--metabolite_name_col")
    parser.add_argument("--compound_feature_prefix")
    parser.add_argument("--metabolite_feature_prefix")
    parser.add_argument("--compound_feature_cols", help="Comma-separated columns.")
    parser.add_argument("--metabolite_feature_cols", help="Comma-separated columns.")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc. Default: auto.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--save_fold_probs",
        dest="save_fold_probs",
        action="store_true",
        default=True,
        help="Save per-fold probability columns. Default: enabled.",
    )
    parser.add_argument(
        "--no_save_fold_probs",
        dest="save_fold_probs",
        action="store_false",
        help="Do not save per-fold probability columns.",
    )
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="Input delimiter. Use auto, comma, tab, or a literal delimiter.",
    )
    parser.add_argument(
        "--model_glob",
        default="*.pt",
        help="Glob under --model_dir for ensemble checkpoint files.",
    )
    parser.add_argument(
        "--expected_folds",
        type=int,
        default=5,
        help="Expected number of model files. Set 0 to disable the check.",
    )
    parser.add_argument(
        "--allow_random_missing_external_features",
        action="store_true",
        help=(
            "Allow deterministic random features for target compounds/metabolites "
            "missing from both training KPGT tables and input feature columns."
        ),
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional debugging limit; 0 means use all input rows.",
    )

    # Background graph and feature files. Defaults mirror CMI_finetune_260611.py.
    parser.add_argument("--ppi_file", default=DEFAULTS["ppi_file"])
    parser.add_argument("--mpi_file", default=DEFAULTS["mpi_file"])
    parser.add_argument("--gpi_file", default=DEFAULTS["gpi_file"])
    parser.add_argument("--ggi_file", default=DEFAULTS["ggi_file"])
    parser.add_argument("--cgi_file", default=DEFAULTS["cgi_file"])
    parser.add_argument("--cpi_pos_file", default=DEFAULTS["cpi_pos_file"])
    parser.add_argument("--compound_features_file", default=DEFAULTS["compound_features_file"])
    parser.add_argument("--protein_features_file", default=DEFAULTS["protein_features_file"])
    parser.add_argument("--metabolite_features_file", default=DEFAULTS["metabolite_features_file"])
    parser.add_argument("--gene_features_file", default=DEFAULTS["gene_features_file"])
    parser.add_argument(
        "--external_compound_features_file",
        help="Optional extra compound KPGT feature CSV/TSV to merge before prediction.",
    )
    parser.add_argument(
        "--external_metabolite_features_file",
        help="Optional extra metabolite KPGT feature CSV/TSV to merge before prediction.",
    )
    parser.add_argument(
        "--external_compound_feature_id_col",
        help="ID column in --external_compound_features_file. Default: first column/index.",
    )
    parser.add_argument(
        "--external_metabolite_feature_id_col",
        help="ID column in --external_metabolite_features_file. Default: first column/index.",
    )
    parser.add_argument(
        "--external_feature_delimiter",
        default="auto",
        help="Delimiter for extra KPGT feature files. Default: auto.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .predictor import run
    run(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
