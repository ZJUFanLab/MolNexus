from __future__ import annotations

import logging
import os
import pandas as pd
import torch
from .config import CLASS2REL as CGI_CLASS2REL
from .config import LABEL_ORDER as CGI_LABEL_ORDER

CGI_SCORE_BATCH_SIZE = int(os.environ.get("CGI_SCORE_BATCH_SIZE", "50000"))


class CGITrainingScoring:

    def score_cgi_multiclass(self, model, h, cgi_df: pd.DataFrame):
            """
            Compute the score for each of the three relations, CGI-N / CGI-D / CGI-U,
            separately for every compound-gene pair.
            Return logits: [N, 3]
            """
            triplet_list = []

            for rel in CGI_LABEL_ORDER:
                tmp = cgi_df[["head_id", "tail_id"]].copy()
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

    def iter_df_batches(self, df: pd.DataFrame, batch_size: int):
            n = len(df)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                yield start, end, df.iloc[start:end]

    def score_cgi_multiclass_batched_no_grad(self, model, h, cgi_df: pd.DataFrame, batch_size: int):
            """
            Used during validation/prediction.
            Do not retain the computation graph, preventing excessive GPU memory usage.
            Return CPU logits: [N, 3]
            """
            model.eval()
            logits_list = []

            with torch.no_grad():
                for start, end, sub_df in self.iter_df_batches(cgi_df, batch_size):
                    logits_batch = self.score_cgi_multiclass(model, h, sub_df)
                    logits_list.append(logits_batch.detach().cpu())

            return torch.cat(logits_list, dim=0)

    def backward_cgi_loss_batched(self, model, h, cgi_df, labels, criterion, batch_size):
            """
            Used during training.
            Compute the CGI multiclass loss in batches and immediately run backward for each batch,
            preventing the computation graphs for all batches from accumulating and causing CUDA OOM.
            """
            n = len(cgi_df)
            total_loss_value = 0.0

            n_batches = (n + batch_size - 1) // batch_size

            for batch_idx, (start, end, sub_df) in enumerate(
                self.iter_df_batches(cgi_df, batch_size)
            ):
                logits_batch = self.score_cgi_multiclass(model, h, sub_df)
                labels_batch = labels[start:end]

                loss_batch = criterion(logits_batch, labels_batch)

                weight = float(end - start) / float(n)
                scaled_loss = loss_batch * weight

                retain_graph = batch_idx < n_batches - 1
                scaled_loss.backward(retain_graph=retain_graph)

                total_loss_value += float(loss_batch.detach().item()) * weight

                del logits_batch
                del labels_batch
                del loss_batch
                del scaled_loss

            return total_loss_value

    def save_predictions(self, model, h, cgi_df, labels, out_csv):
            model.eval()

            logits_cpu = self.score_cgi_multiclass_batched_no_grad(
                model=model,
                h=h,
                cgi_df=cgi_df,
                batch_size=CGI_SCORE_BATCH_SIZE,
            )

            probs = torch.softmax(logits_cpu, dim=1).numpy()
            pred_ids = probs.argmax(axis=1)
            true_ids = labels.detach().cpu().numpy()

            recs = []
            for i, row in cgi_df.reset_index(drop=True).iterrows():
                rec = {
                    "compound": self.id2ent[int(row["head_id"])],
                    "gene": self.id2ent[int(row["tail_id"])],
                    "true_relation": CGI_CLASS2REL[int(true_ids[i])],
                    "pred_relation": CGI_CLASS2REL[int(pred_ids[i])],
                    "correct": bool(int(true_ids[i]) == int(pred_ids[i])),
                }

                for class_id, rel in CGI_CLASS2REL.items():
                    rec[f"prob_{rel}"] = float(probs[i, class_id])

                recs.append(rec)

            pd.DataFrame(recs).to_csv(out_csv, index=False)
            logging.info(f"Saved {len(recs)} CGI predictions to {out_csv}")
