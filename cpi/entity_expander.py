# entity_expander.py

import numpy as np
import pandas as pd
import torch

from .utils import _read_table


def add_entity_if_missing(
    trainer,
    ent: str,
    etype: str,
    vector_np=None,
):
    """
    If the entity does not exist, append it to:
      - trainer.ent2id
      - trainer.id2ent
      - trainer.ent2type
      - trainer.entity_features

    Returns:
      (entity_id, was_added)
    """
    key = (ent, etype)
    if key in trainer.ent2id:
        return trainer.ent2id[key], False

    new_id = len(trainer.ent2id)
    trainer.ent2id[key] = new_id
    trainer.id2ent[new_id] = ent
    trainer.ent2type[new_id] = etype

    if vector_np is None:
        raise ValueError(f"vector_np must not be None for new entity: {(ent, etype)}")

    trainer.entity_features.append(
        torch.tensor(vector_np, dtype=torch.float32, device=trainer.device)
    )
    return new_id, True


def get_default_random_vector(trainer, etype: str):
    """
    Return a randomly initialized vector by entity type, using the dimensions from the original logic.
    """
    if etype == "compound":
        dim = trainer.cpd_features.shape[1]
    elif etype == "protein":
        dim = trainer.protein_features.shape[1]
    elif etype == "metabolite":
        dim = trainer.metabolite_features.shape[1]
    elif etype == "gene":
        dim = trainer.gene_features.shape[1]
    else:
        raise ValueError(f"Unknown entity type: {etype}")

    return np.random.randn(dim).astype("float32")


def resolve_feature_vector(
    trainer,
    ent: str,
    etype: str,
    external_feature_table=None,
):
    """
    Resolve the feature vector using the original logic:
      1) If external_feature_table is provided and contains ent, use it first
      2) Otherwise, retrieve it from the feature tables bundled with trainer
      3) Otherwise, initialize it randomly

    Return a numpy.float32 vector
    """
    if external_feature_table is not None and ent in external_feature_table.index:
        return external_feature_table.loc[ent].values.astype("float32")

    if etype == "compound":
        if ent in trainer.cpd_features.index:
            return trainer.cpd_features.loc[ent].values.astype("float32")
        return get_default_random_vector(trainer, etype)

    if etype == "protein":
        if ent in trainer.protein_features.index:
            return trainer.protein_features.loc[ent].values.astype("float32")
        return get_default_random_vector(trainer, etype)

    if etype == "metabolite":
        if ent in trainer.metabolite_features.index:
            return trainer.metabolite_features.loc[ent].values.astype("float32")
        return get_default_random_vector(trainer, etype)

    if etype == "gene":
        if ent in trainer.gene_features.index:
            return trainer.gene_features.loc[ent].values.astype("float32")
        return get_default_random_vector(trainer, etype)

    raise ValueError(f"Unknown entity type: {etype}")


def load_external_compound_features(external_cpd_features_file: str | None):
    """
    Load the external compound feature file; return None if it is not provided.
    """
    if not external_cpd_features_file:
        return None

    ext_feats = _read_table(external_cpd_features_file, index_col=0, dtype={0: str})
    ext_feats.index = ext_feats.index.astype(str)
    ext_feats = ext_feats.apply(pd.to_numeric, errors="coerce").astype("float32")
    return ext_feats


def _validate_external_compound_features(trainer, compounds, external_feature_table):
    if external_feature_table is None:
        return

    missing = sorted(
        {
            cpd
            for cpd in compounds
            if (cpd, "compound") not in trainer.ent2id
            and cpd not in external_feature_table.index
        }
    )
    if missing:
        raise ValueError(
            f"Missing external compound features for {len(missing)} compounds "
            f"in the external feature table: {missing[:10]}"
        )


def expand_compounds_from_list(
    trainer,
    compounds: list[str],
    external_feature_table=None,
):
    """
    Add compounds from the compound list that are absent from trainer.ent2id.

    Returns:
      newly_added_compounds: list[str]
    """
    _validate_external_compound_features(trainer, compounds, external_feature_table)
    newly_added = []

    for cpd in compounds:
        key = (cpd, "compound")
        if key in trainer.ent2id:
            continue

        vec_np = resolve_feature_vector(
            trainer,
            cpd,
            "compound",
            external_feature_table=external_feature_table,
        )
        add_entity_if_missing(trainer, cpd, "compound", vec_np)
        newly_added.append(cpd)

    return newly_added


def expand_compounds_and_proteins_from_frames(
    trainer,
    frames,
    external_compound_feature_table=None,
):
    """
    Add missing entries from several dataframes:
      - all compounds that appear
      - all proteins that appear

    The logic corresponds to the original test_external:
      - collection of cpd_need / prot_need
      - adding entries to ent2id / feature one by one

    Parameters:
      frames: list[pd.DataFrame]
    """
    cpd_need = set()
    prot_need = set()

    for df in frames:
        if df is None or df.empty:
            continue

        if "head" in df.columns and "head_type" in df.columns:
            cpd_need |= set(df.loc[df["head_type"] == "compound", "head"].astype(str).tolist())
            prot_need |= set(df.loc[df["head_type"] == "protein", "head"].astype(str).tolist())

        if "tail" in df.columns and "tail_type" in df.columns:
            cpd_need |= set(df.loc[df["tail_type"] == "compound", "tail"].astype(str).tolist())
            prot_need |= set(df.loc[df["tail_type"] == "protein", "tail"].astype(str).tolist())

    _validate_external_compound_features(trainer, cpd_need, external_compound_feature_table)

    for cpd in cpd_need:
        if (cpd, "compound") in trainer.ent2id:
            continue
        vec_np = resolve_feature_vector(
            trainer,
            cpd,
            "compound",
            external_feature_table=external_compound_feature_table,
        )
        add_entity_if_missing(trainer, cpd, "compound", vec_np)

    for prot in prot_need:
        if (prot, "protein") in trainer.ent2id:
            continue
        vec_np = resolve_feature_vector(trainer, prot, "protein", external_feature_table=None)
        add_entity_if_missing(trainer, prot, "protein", vec_np)


def expand_entities_from_cgi_frame(
    trainer,
    cgi_df: pd.DataFrame,
):
    """
    Add all entity types appearing in the external CGI dataframe.
    The logic corresponds to the original unlabeled prediction:
      - consistent addition of compound / protein / metabolite / gene entries

    Returns:
      added_by_type: dict[str, list[str]]
    """
    added_by_type = {
        "compound": [],
        "protein": [],
        "metabolite": [],
        "gene": [],
    }

    if cgi_df is None or cgi_df.empty:
        return added_by_type

    for ecol, tcol in [("head", "head_type"), ("tail", "tail_type")]:
        if ecol not in cgi_df.columns or tcol not in cgi_df.columns:
            continue

        for ent, etype in cgi_df[[ecol, tcol]].astype(str).itertuples(index=False):
            key = (ent, etype)
            if key in trainer.ent2id:
                continue

            vec_np = resolve_feature_vector(trainer, ent, etype, external_feature_table=None)
            _, was_added = add_entity_if_missing(trainer, ent, etype, vec_np)
            if was_added:
                added_by_type.setdefault(etype, []).append(ent)

    return added_by_type


def expand_candidate_proteins_from_file(
    trainer,
    candidate_file: str,
):
    """
    Load protein candidates from candidate_file and add missing proteins.
    The logic corresponds to the add-protein section in the original _cands_from_file.

    Returns:
      protein_ids: list[int]
      raw_count: int
      unique_count: int
    """
    dfc = _read_table(candidate_file, dtype=str)
    if dfc is None or dfc.empty:
        raise ValueError(f"candidate_file is empty: {candidate_file}")

    if "tail" in dfc.columns:
        if "tail_type" in dfc.columns:
            tails = dfc.loc[dfc["tail_type"].astype(str).str.lower().eq("protein"), "tail"].astype(str)
        else:
            tails = dfc["tail"].astype(str)
    elif "protein" in dfc.columns:
        tails = dfc["protein"].astype(str)
    else:
        raise ValueError("candidate_file must contain 'tail' column (optionally filter by 'tail_type'=='protein') or the 'protein' column")

    uniq = sorted(set(tails.tolist()))
    ids = []

    for prot in uniq:
        key = (prot, "protein")
        if key not in trainer.ent2id:
            vec_np = resolve_feature_vector(trainer, prot, "protein", external_feature_table=None)
            add_entity_if_missing(trainer, prot, "protein", vec_np)
        ids.append(trainer.ent2id[key])

    if len(ids) == 0:
        raise ValueError(f"No protein IDs were extracted from candidate_file: {candidate_file}")

    return ids, len(tails), len(uniq)
