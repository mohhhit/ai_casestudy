"""
================================================================================
 Snap Inc. (Snapchat) — AI Deployment Strategy
 Module: models/churn_predictor.py
 Purpose: Train & evaluate a Gradient Boosted Tree (XGBoost) churn predictor
          with SMOTE oversampling, threshold tuning, and SHAP-style
          bar-chart feature importance.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, f1_score,
    ConfusionMatrixDisplay, RocCurveDisplay
)

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLS = ["age_group", "country"]
FEATURE_COLS     = ["tenure_days", "age_group", "country", "avg_daily_opens",
                    "avg_snaps_sent", "avg_stories_posted", "streak_max",
                    "streak_current", "friend_count", "notifications_on",
                    "used_spotlight", "used_bitmoji", "premium_features",
                    "days_since_last_open"]
TARGET_COL       = "churned"


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


def find_best_threshold(y_true, y_prob, metric="f1") -> float:
    """Sweep thresholds 0.1–0.9 and return the one maximising the chosen metric."""
    best_thresh, best_score = 0.5, 0.0
    for t in np.arange(0.10, 0.91, 0.01):
        y_hat = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_hat, zero_division=0)
        if score > best_score:
            best_score, best_thresh = score, t
    return round(best_thresh, 2)


def train_churn_model(df: pd.DataFrame) -> dict:
    """
    Train XGBoost churn predictor with SMOTE oversampling and threshold tuning.

    Returns
    -------
    dict with model, metrics, optimal threshold, feature importance.
    """
    print("\n" + "=" * 60)
    print("  MODEL 3 — User Churn Prediction  (XGBoost + SMOTE)")
    print("=" * 60)

    df_enc, _ = preprocess(df)
    X = df_enc[FEATURE_COLS]
    y = df_enc[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # SMOTE on train set only
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"\n  After SMOTE | Class 0: {(y_res==0).sum()}  Class 1: {(y_res==1).sum()}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_res, y_res,
              eval_set=[(X_test, y_test)],
              verbose=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring="roc_auc", n_jobs=-1)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    roc_auc     = roc_auc_score(y_test, y_pred_prob)

    # Threshold tuning
    best_thresh = find_best_threshold(y_test, y_pred_prob)
    y_pred_opt  = (y_pred_prob >= best_thresh).astype(int)

    report = classification_report(y_test, y_pred_opt, output_dict=True)

    print(f"\n  CV ROC-AUC      : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Test ROC-AUC    : {roc_auc:.4f}")
    print(f"  Optimal Threshold: {best_thresh}")
    print(f"\n{classification_report(y_test, y_pred_opt)}")

    fi_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    _plot_churn_results(model, X_test, y_test, y_pred_prob, y_pred_opt, fi_df, best_thresh)

    return {
        "model":                 model,
        "cv_roc_auc":            cv_scores.mean(),
        "test_roc_auc":          roc_auc,
        "optimal_threshold":     best_thresh,
        "classification_report": report,
        "feature_importance":    fi_df,
        "X_test":                X_test,
        "y_test":                y_test,
        "y_pred_proba":          y_pred_prob,
    }


def _plot_churn_results(model, X_test, y_test, y_pred_prob, y_pred_opt, fi_df, thresh):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Snap Inc. — User Churn Predictor (threshold={thresh})", fontsize=14, fontweight="bold")

    # Confusion Matrix at optimal threshold
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_opt, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (t={thresh})")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_pred_prob, ax=axes[1])
    axes[1].get_lines()[0].set_color("crimson")
    axes[1].plot([0, 1], [0, 1], "k--")
    axes[1].set_title("ROC Curve")

    # Feature Importance
    sns.barplot(data=fi_df.head(10), x="importance", y="feature", ax=axes[2], palette="crest")
    axes[2].set_title("Top-10 Feature Importances")
    axes[2].set_xlabel("XGBoost Gain")

    plt.tight_layout()
    out_path = REPORTS_DIR / "churn_model_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Plot saved] {out_path}")


if __name__ == "__main__":
    from data.data_generator import generate_churn_data
    df = generate_churn_data()
    train_churn_model(df)

