from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
SCRIPT_DIR = PACKAGE_DIR
PREDICT_CODE_DIR = PACKAGE_DIR
# CMI fine-tuning always uses the global CGI background.
CGI_GLOBAL = "CGI_GLOBAL"
TRAINING_SCRIPT = PACKAGE_DIR / "trainer.py"
SEED = 42
LABEL_ORDER = ["CMI-N", "CMI-D", "CMI-U"]
REL2CLASS = {rel: i for i, rel in enumerate(LABEL_ORDER)}
CLASS2REL = {i: rel for rel, i in REL2CLASS.items()}
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HGT_EMB_DIM = 512
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_NUM_LAYERS = 1
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_TRAIN_CKPT_DIR = "result_CMI"
DEFAULT_RUN_TAG = "CMI_finetune"
TRAIN_PATHS = {
    "train_template": "demo/task_demo/cmi/fold{fold}_train.csv",
    "val_template": "demo/task_demo/cmi/fold{fold}_val.csv",
}
DEFAULTS = {
    "model_dir": str(PREDICT_CODE_DIR / "model"),
    "output_dir": str(PREDICT_CODE_DIR / "output"),
    "ppi_file": "demo/BioNexKG_demo/ppi.csv",
    "mpi_file": "demo/BioNexKG_demo/mpi.csv",
    "gpi_file": "demo/BioNexKG_demo/gpi.csv",
    "ggi_file": "demo/BioNexKG_demo/ggi.csv",
    "cgi_file": "demo/BioNexKG_demo/cgi.csv",
    "cpi_pos_file": "demo/BioNexKG_demo/cpi.csv",
    "compound_features_file": "demo/feat_demo_subset/demo_compounds_feat.csv",
    "protein_features_file": "demo/feat_demo_subset/demo_proteins_feat.csv",
    "metabolite_features_file": "demo/feat_demo_subset/demo_metabolites_feat.csv",
    "gene_features_file": "demo/feat_demo_subset/demo_genes_feat.csv",
}

LABEL_CANDIDATES = [
    "label",
    "relation",
    "class",
    "y",
    "target",
    "true_label",
    "true_relation",
    "rel",
]

COMMON_META_COLUMNS = {
    "head",
    "tail",
    "head_type",
    "tail_type",
    "compound",
    "compound_id",
    "cid",
    "CID",
    "cpd",
    "cpd_id",
    "drug",
    "drug_id",
    "metabolite",
    "metabolite_id",
    "hmdb",
    "hmdb_id",
    "compound_name",
    "metabolite_name",
    "name",
}
