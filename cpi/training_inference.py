from __future__ import annotations

from .predictor import (
    predict_unlabeled_all_impl,
    predict_unlabeled_topk_impl,
    save_predictions_impl,
    test_external_impl,
)


class CPITrainingInference:

    def save_predictions(self, model, h, triplets, labels, out_csv, threshold=0.5):
            return save_predictions_impl(self, model, h, triplets, labels, out_csv, threshold)

    def test_external(
            self,
            external_cgi_file: str,
            external_cpi_file: str,
            external_cpd_features_file: str,
            checkpoint_path: str,
            ablation_mode: str | None = None,
            cgi_mode: str | None = None,
        ):
            return test_external_impl(
                self,
                external_cgi_file,
                external_cpi_file,
                external_cpd_features_file,
                checkpoint_path,
                ablation_mode,
                cgi_mode,
            )

    def predict_unlabeled_topk(
            self,
            compound_file: str,
            external_cpd_features_file: str | None,
            checkpoint_path: str,
            candidate_file: str,
            topk: int = 20,
            ablation_mode: str | None = None,
            cgi_mode: str | None = None,
            external_cgi_file: str | None = None,
            batch_size: int = 131072,
            chunk_size: int = 32768,
        ):
            return predict_unlabeled_topk_impl(
                self,
                compound_file,
                external_cpd_features_file,
                checkpoint_path,
                candidate_file,
                topk,
                ablation_mode,
                cgi_mode,
                external_cgi_file,
                batch_size,
                chunk_size,
            )

    def predict_unlabeled_all(
            self,
            compound_file: str,
            external_cpd_features_file: str | None,
            checkpoint_path: str,
            candidate_file: str,
            ablation_mode: str | None = None,
            cgi_mode: str | None = None,
            external_cgi_file: str | None = None,
            batch_size: int = 131072,
            chunk_size: int = 32768,
        ):
            return predict_unlabeled_all_impl(
                self,
                compound_file,
                external_cpd_features_file,
                checkpoint_path,
                candidate_file,
                ablation_mode,
                cgi_mode,
                external_cgi_file,
                batch_size,
                chunk_size,
            )
