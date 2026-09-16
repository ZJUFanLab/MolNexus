from __future__ import annotations

import logging
import pandas as pd
import torch
from .config import CLASS2REL as CMI_CLASS2REL
from .config import LABEL_ORDER as CMI_LABEL_ORDER


class CMITrainingScoring:

    def score_cmi_multiclass(self, model, h, cmi_df: pd.DataFrame):
            """
            Compute the score for each of the three relations, CMI-N / CMI-D / CMI-U,
            separately for every compound-metabolite pair.
            Return logits: [N, 3]
            """
            triplet_list = []

            for rel in CMI_LABEL_ORDER:
                tmp = cmi_df[["head_id", "tail_id"]].copy()
                tmp["relation_id"] = self.rel2id[rel]
                arr = tmp[["head_id", "relation_id", "tail_id"]].values.astype(int)
                triplet_list.append(
                    torch.from_numpy(arr).long().to(self.device)
                )

            logits = []
            for triplets in triplet_list:
                score = model.get_score(h, triplets)
                logits.append(score)

            logits = torch.stack(logits, dim=1)

            return logits

    def save_predictions(self, model, h, cmi_df, labels, out_csv):
            model.eval()
            with torch.no_grad():
                logits = self.score_cmi_multiclass(model, h, cmi_df)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                pred_ids = probs.argmax(axis=1)
                true_ids = labels.detach().cpu().numpy()

                recs = []
                for i, row in cmi_df.reset_index(drop=True).iterrows():
                    rec = {
                        "compound": self.id2ent[int(row["head_id"])],
                        "metabolite": self.id2ent[int(row["tail_id"])],
                        "true_relation": CMI_CLASS2REL[int(true_ids[i])],
                        "pred_relation": CMI_CLASS2REL[int(pred_ids[i])],
                        "correct": bool(int(true_ids[i]) == int(pred_ids[i])),
                    }

                    for class_id, rel in CMI_CLASS2REL.items():
                        rec[f"prob_{rel}"] = float(probs[i, class_id])

                    recs.append(rec)

                pd.DataFrame(recs).to_csv(out_csv, index=False)
                logging.info(f"Saved {len(recs)} CMI predictions to {out_csv}")
