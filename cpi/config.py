# config.py

ALLOWED_TYPES = {"compound", "protein", "metabolite", "gene"}

# CGI graph construction modes used by CPI pretraining and inference.
CGI_GLOBAL = "CGI_GLOBAL"
CGI_SPLIT = "CGI_SPLIT"

ABLATION_RELATIONS = {
    "FULL": frozenset({"CGI", "GGI", "GPI", "PPI", "MPI"}),
    "NO_CGI": frozenset({"GGI", "GPI", "PPI", "MPI"}),
    "NO_GGI": frozenset({"CGI", "GPI", "PPI", "MPI"}),
    "NO_GPI": frozenset({"CGI", "GGI", "PPI", "MPI"}),
    "NO_PPI": frozenset({"CGI", "GGI", "GPI", "MPI"}),
    "NO_MPI": frozenset({"CGI", "GGI", "GPI", "PPI"}),
    "NO_PROTEIN_RELATED": frozenset({"CGI", "GGI"}),
    "NO_GENE_RELATED": frozenset({"PPI", "MPI"}),
    "CPI_ONLY": frozenset(),
}
ABLATION_MODES = frozenset(ABLATION_RELATIONS)

def normalize_ablation_mode(ablation_mode: str) -> str:
    """Return the canonical ablation name or raise for an unsupported value."""
    if ablation_mode not in ABLATION_MODES:
        supported = ", ".join(sorted(ABLATION_MODES))
        raise ValueError(
            f"Unknown ablation mode: {ablation_mode!r}. "
            f"Supported canonical modes: {supported}"
        )
    return ablation_mode

SUPPORTED_TRAIN_MODES = {"Warm", "CCS", "PCS", "DCS"}

DEFAULT_TRAIN_MODE = "Warm"
DEFAULT_PREDICT_SPLIT = DEFAULT_TRAIN_MODE
DEFAULT_ENSEMBLE_SIZE = 5
DEFAULT_MODEL_ROOT = "model"


DEFAULT_LOG_FILE = "training.log"

DEFAULT_SUMMARY_PREFIX = ""

DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HGT_EMB_DIM = 512
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_NUM_LAYERS = 1
DEFAULT_NUM_EPOCHS = 500
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_MLP_HIDDEN_DIM = 512
DEFAULT_HGT_DROPOUT = 0.2
DEFAULT_MLP_DROPOUT = 0.2

TRAIN_MODE_CONFIGS = {
    "Warm": {
        "learning_rate": 1e-3,
        "hgt_emb_dim": 512,
        "num_heads": 4,
        "num_layers": 1,
        "num_epochs": 500,
        "weight_decay": 1e-3,
        "mlp_hidden_dim": 512,
        "hgt_dropout": 0.5,
        "mlp_dropout": 0.5,
        "ckpt_dir": "result_Warm",
        "log_file": "training_Warm.log",
        "summary_prefix": "Warm",
        "train_csv_template": "task_demo/cpi/Warm/fold{fold}_train.csv",
        "val_csv_template": "task_demo/cpi/Warm/fold{fold}_val.csv",
    },
    "CCS": {
        "learning_rate": 1e-3,
        "hgt_emb_dim": 512,
        "num_heads": 4,
        "num_layers": 1,
        "num_epochs": 100,
        "weight_decay": 0,
        "mlp_hidden_dim": 512,
        "hgt_dropout": 0.2,
        "mlp_dropout": 0.2,
        "ckpt_dir": "model/CCS",
        "log_file": "training_CCS.log",
        "summary_prefix": "CCS",
        "train_csv_template": "task_demo/cpi/CCS/fold{fold}_train.csv",
        "val_csv_template": "task_demo/cpi/CCS/fold{fold}_val.csv",
    },
    "PCS": {
        "learning_rate": 3e-4,
        "hgt_emb_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "num_epochs": 100,
        "weight_decay": 5e-4,
        "mlp_hidden_dim": 128,
        "hgt_dropout": 0.2,
        "mlp_dropout": 0.2,
        "ckpt_dir": "result_PCS",
        "log_file": "training_PCS.log",
        "summary_prefix": "PCS",
        "train_csv_template": "task_demo/cpi/PCS/fold{fold}_train.csv",
        "val_csv_template": "task_demo/cpi/PCS/fold{fold}_val.csv",
    },
    "DCS": {
        "learning_rate": 1e-3,
        "hgt_emb_dim": 128,
        "num_heads": 2,
        "num_layers": 1,
        "num_epochs": 100,
        "weight_decay": 1e-2,
        "mlp_hidden_dim": 128,
        "hgt_dropout": 0.3,
        "mlp_dropout": 0.3,
        "ckpt_dir": "result_DCS",
        "log_file": "training_DCS.log",
        "summary_prefix": "DCS",
        "train_csv_template": "task_demo/cpi/DCS/fold{fold}_train.csv",
        "val_csv_template": "task_demo/cpi/DCS/fold{fold}_val.csv",
    },
}

DEFAULT_SEED = 42
