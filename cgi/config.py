from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
SEED = 42
LABEL_ORDER = ["CGI-N", "CGI-D", "CGI-U"]
REL2CLASS = {rel: i for i, rel in enumerate(LABEL_ORDER)}
CLASS2REL = {i: rel for rel, i in REL2CLASS.items()}
NODE_TYPES = ["compound", "metabolite", "protein", "gene"]
TYPE_ORDER = ["compound", "metabolite", "protein", "gene"]
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_HGT_EMB_DIM = 512
DEFAULT_NUM_HEADS = 4
DEFAULT_DROPOUT = 0.2
DEFAULT_NUM_LAYERS = 1
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_SCORE_BATCH_SIZE = 50000
DEFAULT_TRAIN_CKPT_DIR = "result_CGI"
DEFAULT_RUN_TAG = "CGI_finetune"
TRAINING_SCRIPT = str(PACKAGE_DIR / "trainer.py")

DEFAULT_PATHS = {
    "train_template": "demo/task_demo/cgi/fold{fold}_train.csv",
    "val_template": "demo/task_demo/cgi/fold{fold}_val.csv",
    "ppi_file": "demo/BioNexKG_demo/ppi.csv",
    "mpi_file": "demo/BioNexKG_demo/mpi.csv",
    "gpi_file": "demo/BioNexKG_demo/gpi.csv",
    "ggi_file": "demo/BioNexKG_demo/ggi.csv",
    "cpi_pos_file": "demo/BioNexKG_demo/cpi.csv",
    "cpd_features_file": "demo/feat_demo_subset/demo_compounds_feat.csv",
    "protein_features_file": "demo/feat_demo_subset/demo_proteins_feat.csv",
    "metabolite_features_file": "demo/feat_demo_subset/demo_metabolites_feat.csv",
    "gene_features_file": "demo/feat_demo_subset/demo_genes_feat.csv",
}

COMMON_COMPOUND_ID_COLS = ["compound_id", "compound", "cid", "CID", "cpd_id", "drug_id", "pert_id", "head"]
COMMON_GENE_ID_COLS = ["gene_id", "gene", "gene_symbol", "symbol", "target_gene", "tail"]
COMMON_COMPOUND_NAME_COLS = ["compound_name", "cpd_name", "drug_name", "pert_name", "name"]
COMMON_GENE_NAME_COLS = ["gene_name", "target_name"]
COMMON_LABEL_COLS = ["label", "relation", "class", "y", "target", "true_label", "true_relation"]
COMMON_COMPOUND_FEATURE_PREFIXES = ["compound_kpgt_", "cpd_kpgt_", "compound_feature_", "cpd_feature_", "compound_feat_"]
COMMON_GENE_FEATURE_PREFIXES = ["gene_kpgt_", "gene_feature_", "gene_feat_", "gene_esm_"]
