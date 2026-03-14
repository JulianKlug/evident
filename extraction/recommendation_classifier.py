"""ML-based recommendation classifier: BioLORD embeddings + logistic regression.

Trains a binary classifier to distinguish true recommendations (TP) from
non-recommendations (FP) extracted by the LLM pipeline.  Uses leave-one-
guideline-out cross-validation (LOGO-CV) for honest evaluation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def train_classifier(
    data_csv: str = "artifacts/classifier/training_data.csv",
    embeddings_npz: str = "artifacts/classifier/embeddings.npz",
    model_output: str = "artifacts/classifier/rec_classifier.joblib",
    C: float = 1.0,
) -> tuple:
    """Train logistic regression on BioLORD embeddings with LOGO-CV.

    Returns:
        (model, cv_results_df) where cv_results_df has per-guideline P/R/F1
        computed on held-out predictions.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_score, recall_score, f1_score
    import joblib

    df = pd.read_csv(data_csv)
    data = np.load(embeddings_npz)
    X = data["embeddings"]
    y = df["label"].values
    guideline_keys = df["guideline_key"].values

    unique_keys = sorted(set(guideline_keys))
    print(f"[Classifier] Training on {len(df)} examples, {len(unique_keys)} guidelines")
    print(f"[Classifier] Class distribution: {(y == 1).sum()} TP, {(y == 0).sum()} FP")

    # LOGO-CV: leave-one-guideline-out
    held_out_preds = np.zeros(len(y), dtype=int)
    held_out_probs = np.zeros(len(y))
    cv_results = []

    for key in unique_keys:
        test_mask = guideline_keys == key
        train_mask = ~test_mask

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            continue

        clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
        clf.fit(X[train_mask], y[train_mask])
        preds = clf.predict(X[test_mask])
        probs = clf.predict_proba(X[test_mask])[:, 1]
        held_out_preds[test_mask] = preds
        held_out_probs[test_mask] = probs

        y_test = y[test_mask]
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        n_kept = preds.sum()
        n_total = len(preds)
        cv_results.append({
            "guideline_key": key,
            "n_total": n_total,
            "n_tp": int(y_test.sum()),
            "n_fp": int((y_test == 0).sum()),
            "n_kept": int(n_kept),
            "n_removed": int(n_total - n_kept),
            "precision": p,
            "recall": r,
            "f1": f,
        })
        print(f"  [{key}] P={p:.3f} R={r:.3f} F1={f:.3f} "
              f"(kept {n_kept}/{n_total}, {y_test.sum()} TP, {(y_test == 0).sum()} FP)")

    cv_df = pd.DataFrame(cv_results)
    avg_p = precision_score(y, held_out_preds, zero_division=0)
    avg_r = recall_score(y, held_out_preds, zero_division=0)
    avg_f = f1_score(y, held_out_preds, zero_division=0)
    print(f"\n[Classifier] LOGO-CV Overall: P={avg_p:.3f} R={avg_r:.3f} F1={avg_f:.3f}")
    print(f"[Classifier] Kept {held_out_preds.sum()}/{len(held_out_preds)} "
          f"({held_out_preds.sum()/len(held_out_preds)*100:.1f}%)")

    # Train final model on all data
    final_clf = LogisticRegression(C=C, max_iter=1000, class_weight="balanced")
    final_clf.fit(X, y)
    joblib.dump(final_clf, model_output)
    print(f"[Classifier] Final model saved to {model_output}")

    # Save held-out predictions for downstream LOGO-CV evaluation
    df["logo_pred"] = held_out_preds
    df["logo_prob"] = held_out_probs
    logo_path = data_csv.replace(".csv", "_logo_preds.csv")
    df.to_csv(logo_path, index=False)
    print(f"[Classifier] LOGO-CV predictions saved to {logo_path}")

    return final_clf, cv_df


def load_classifier(model_path: str = "artifacts/classifier/rec_classifier.joblib"):
    """Load a trained classifier from disk."""
    import joblib
    return joblib.load(model_path)


def ml_filter_recommendations(
    df: pd.DataFrame,
    classifier,
    similarity_model,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """Filter extracted recommendations using the ML classifier.

    Args:
        df: DataFrame with 'recommendation' column.
        classifier: Trained sklearn classifier with .predict_proba().
        similarity_model: Model with encode_batch() for BioLORD embeddings.
        threshold: Probability threshold for keeping a recommendation.
            Default 0.3 (conservative — removes only high-confidence non-recs).

    Returns:
        Filtered DataFrame containing only predicted positive recommendations.
    """
    if df.empty:
        return df

    texts = df["recommendation"].tolist()
    embeddings = similarity_model.encode_batch(texts)
    probs = classifier.predict_proba(embeddings)[:, 1]
    keep_mask = probs >= threshold

    n_kept = int(keep_mask.sum())
    n_removed = len(keep_mask) - n_kept
    print(f"  [ML Filter] Kept {n_kept}/{len(keep_mask)} recommendations "
          f"(removed {n_removed} non-recs, threshold={threshold})", flush=True)

    filtered = df[keep_mask].reset_index(drop=True)
    return filtered
