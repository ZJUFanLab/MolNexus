from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch


class CPITrainingData:

    def create_entity_features(self):
            num_entities = len(self.ent2id)
            features_list = []

            tables = {
                "compound": (self.cpd_features, self.cpd_features.shape[1]),
                "metabolite": (self.metabolite_features, self.metabolite_features.shape[1]),
                "protein": (self.protein_features, self.protein_features.shape[1]),
                "gene": (self.gene_features, self.gene_features.shape[1]),
            }

            for idx in range(num_entities):
                ent = self.id2ent[idx]
                etype = self.ent2type[idx]
                df, dim = tables.get(etype, (None, None))

                if df is not None and ent in df.index:
                    arr = df.loc[ent].values
                else:
                    logging.warning(f"Entity '{ent}' of type '{etype}' has no features; initializing randomly")
                    arr = np.random.randn(dim)

                features_list.append(torch.tensor(arr, dtype=torch.float32, device=self.device))

            return features_list

    def map_triplets(self, df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            df["key_head"] = list(zip(df["head"], df["head_type"]))
            df["key_tail"] = list(zip(df["tail"], df["tail_type"]))
            df["head_id"] = df["key_head"].map(self.ent2id)
            df["tail_id"] = df["key_tail"].map(self.ent2id)
            df = df.dropna(subset=["head_id", "tail_id"])
            df[["head_id", "tail_id"]] = df[["head_id", "tail_id"]].astype(int)
            df["relation_id"] = df["relation"].map(self.rel2id)
            return df

    def build_train_val_test_data(self):
            train_data = pd.concat(
                [self.train_df, self.ppi_df, self.mpi_df, self.gpi_df, self.ggi_df, self.cgi_df],
                ignore_index=True,
            )
            val_data = pd.concat(
                [self.val_df, self.ppi_df, self.mpi_df, self.gpi_df, self.ggi_df, self.cgi_df],
                ignore_index=True,
            )
            train_data = self.map_triplets(train_data)
            val_data = self.map_triplets(val_data)
            logging.info(f"Train triplets: {train_data.head()}")
            logging.info(f"Validation triplets: {val_data.head()}")
            train_cpd = train_data[train_data["relation_id"] == self.rel2id["CPI"]]
            val_cpd = val_data[val_data["relation_id"] == self.rel2id["CPI"]]
            return train_data, val_data, train_cpd, val_cpd
