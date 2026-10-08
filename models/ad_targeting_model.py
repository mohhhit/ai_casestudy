"""
================================================================================
 Snap Inc. (Snapchat) — AI Deployment Strategy
 Module: models/ad_targeting_model.py
 Purpose: Train & evaluate a Gradient Boosting (XGBoost) model for
          programmatic ad click-through-rate (CTR) prediction.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    ConfusionMatrixDisplay, PrecisionRecallDisplay
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["user_segment", "ad_format", "ad_category", "device_type", "placement", "user_age_bucket"]
FEATURE_COLS     = ["user_segment", "ad_format", "ad_category", "bid_price_usd", "hour_of_day",
                    "day_of_week", "device_type", "placement", "user_age_bucket", "frequency", "relevance_score"]
TARGET_COL       = "clicked"          # Primary target: CTR


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def train_ad_model(df: pd.DataFrame) -> dict:
    """
    Train XGBoost classifier for ad click-through rate prediction.

    Returns
    -------
    dict with model, metrics, feature importance, and test predictions.
    """
    print("\n" + "=" * 60)
    print("  MODEL 2 — Programmatic Ad Targeting  (XGBoost)")
    print("=" * 60)

    df_enc, _ = preprocess(df)
    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Class imbalance ratio for scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = round(neg_count / max(pos_count, 1), 2)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_pred_prob)
    avg_prec    = average_precision_score(y_test, y_pred_prob)

    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  Scale pos weight : {scale_pos}")
    print(f"  CV ROC-AUC       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test ROC-AUC     : {roc_auc:.4f}")
    print(f"  Avg Precision    : {avg_prec:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    fi_df = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    _plot_ad_results(model, X_test, y_test, y_pred_prob, fi_df)

    return {
        "model":                 model,
        "cv_roc_auc":            cv_scores.mean(),
        "test_roc_auc":          roc_auc,
        "avg_precision":         avg_prec,
        "classification_report": report,
        "feature_importance":    fi_df,
        "X_test":                X_test,
        "y_test":                y_test,
        "y_pred_proba":          y_pred_prob,
    }


def _plot_ad_results(model, X_test, y_test, y_pred_prob, fi_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Snap Inc. — Programmatic Ad Targeting Model (XGBoost)", fontsize=14, fontweight="bold")

    # Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), ax=axes[0])
    axes[0].set_title("Confusion Matrix (CTR)")

    # Precision-Recall Curve (more informative for imbalanced CTR)
    PrecisionRecallDisplay.from_predictions(y_test, y_pred_prob, ax=axes[1], color="steelblue")
    axes[1].set_title("Precision-Recall Curve")

    # Feature Importance
    sns.barplot(data=fi_df.head(8), x="importance", y="feature", ax=axes[2], palette="magma")
    axes[2].set_title("Top-8 Feature Importances")
    axes[2].set_xlabel("XGBoost Gain")

    plt.tight_layout()
    out_path = REPORTS_DIR / "ad_targeting_model_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Plot saved] {out_path}")


if __name__ == "__main__":
    from data.data_generator import generate_ad_targeting_data
    df = generate_ad_targeting_data()
    train_ad_model(df)

