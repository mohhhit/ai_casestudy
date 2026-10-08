"""
================================================================================
 Snap Inc. (Snapchat) — AI Deployment Strategy
 Module: models/ar_lens_recommender.py
 Purpose: Train & evaluate an AR Lens Recommendation model using
          Random Forest (interpretable) with feature-importance analysis.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["age_group", "gender", "country", "lens_category", "time_of_day", "device_type"]
FEATURE_COLS     = ["age_group", "gender", "country", "daily_snaps_sent", "story_views",
                    "session_minutes", "lens_category", "time_of_day", "device_type",
                    "lens_used_cnt", "friend_count", "clicked_discover"]
TARGET_COL       = "engaged_with_ar"


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categoricals; return encoded df and encoder map."""
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def train_ar_model(df: pd.DataFrame) -> dict:
    """
    Train Random Forest classifier for AR lens engagement prediction.

    Returns
    -------
    dict with keys: model, metrics, feature_importance, X_test, y_test, y_pred_proba
    """
    print("\n" + "=" * 60)
    print("  MODEL 1 — AR Lens Recommendation  (Random Forest)")
    print("=" * 60)

    df_enc, _ = preprocess(df)
    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Cross-validated ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_pred_prob)

    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  CV ROC-AUC  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test ROC-AUC: {roc_auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    # Feature importance
    fi_df = pd.DataFrame({
        "feature":   FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    _plot_ar_results(model, X_test, y_test, y_pred_prob, fi_df)

    return {
        "model":              model,
        "cv_roc_auc":         cv_scores.mean(),
        "test_roc_auc":       roc_auc,
        "classification_report": report,
        "feature_importance": fi_df,
        "X_test":             X_test,
        "y_test":             y_test,
        "y_pred_proba":       y_pred_prob,
    }


def _plot_ar_results(model, X_test, y_test, y_pred_prob, fi_df: pd.DataFrame):
    """Generate and save diagnostic plots for the AR model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Snap Inc. — AR Lens Recommendation Model", fontsize=14, fontweight="bold")

    # -- Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(y_test, model.predict(X_test), ax=axes[0])
    axes[0].set_title("Confusion Matrix")

    # -- ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=axes[1])
    axes[1].get_lines()[0].set_color("darkorange")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("ROC Curve")

    # -- Feature Importance
    sns.barplot(data=fi_df.head(8), x="importance", y="feature", ax=axes[2], palette="viridis")
    axes[2].set_title("Top-8 Feature Importances")
    axes[2].set_xlabel("Mean Decrease in Impurity")

    plt.tight_layout()
    out_path = REPORTS_DIR / "ar_lens_model_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Plot saved] {out_path}")


if __name__ == "__main__":
    from data.data_generator import generate_ar_lens_data
    df = generate_ar_lens_data()
    train_ar_model(df)

