from __future__ import annotations

import argparse
from .config import DEFAULT_PATHS, TRAINING_SCRIPT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Predict external compound-gene CGI-N/CGI-D/CGI-U relations "
            "with the HGT 5-fold ensemble."
        )
    )
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--model_dir", default="model")
    parser.add_argument(
        "--input_format",
        choices=["triplet", "kpgt_features", "auto"],
        default="auto",
    )
    parser.add_argument("--label_col", default=None)
    parser.add_argument("--compound_id_col", default=None)
    parser.add_argument("--gene_id_col", default=None)
    parser.add_argument("--compound_name_col", default=None)
    parser.add_argument("--gene_name_col", default=None)
    parser.add_argument("--compound_feature_prefix", default=None)
    parser.add_argument("--gene_feature_prefix", default=None)
    parser.add_argument("--compound_feature_cols", default=None)
    parser.add_argument("--gene_feature_cols", default=None)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--save_fold_probs",
        action="store_true",
        default=True,
        help="Save fold-level probability columns in predictions.csv.",
    )
    parser.add_argument(
        "--no_save_fold_probs",
        dest="save_fold_probs",
        action="store_false",
        help="Do not save fold-level probability columns.",
    )
    parser.add_argument(
        "--delimiter",
        default="auto",
        help="Input delimiter. Use 'auto', ',', 'tab', or '\\t'.",
    )

    parser.add_argument("--training_script", default=TRAINING_SCRIPT)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--decoder", default=None)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--allow_missing_features_random", action="store_true")
    parser.set_defaults(ablation_mode="FULL")
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--ppi_file", default=DEFAULT_PATHS["ppi_file"])
    parser.add_argument("--mpi_file", default=DEFAULT_PATHS["mpi_file"])
    parser.add_argument("--gpi_file", default=DEFAULT_PATHS["gpi_file"])
    parser.add_argument("--ggi_file", default=DEFAULT_PATHS["ggi_file"])
    parser.add_argument("--cpi_pos_file", default=DEFAULT_PATHS["cpi_pos_file"])
    parser.add_argument("--cpd_features_file", default=DEFAULT_PATHS["cpd_features_file"])
    parser.add_argument("--protein_features_file", default=DEFAULT_PATHS["protein_features_file"])
    parser.add_argument("--metabolite_features_file", default=DEFAULT_PATHS["metabolite_features_file"])
    parser.add_argument("--gene_features_file", default=DEFAULT_PATHS["gene_features_file"])
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from .predictor import run
    return run(args)

if __name__ == "__main__":
    raise SystemExit(main())
