"""
================================================================================
 Snap Inc. (Snapchat) — AI Deployment Strategy
 Module: pipeline/run_pipeline.py
 Purpose: End-to-end orchestration script that:
          1. Generates all synthetic datasets
          2. Trains all three AI models
          3. Runs ROI analysis
          4. Generates a consolidated text report
================================================================================
"""

import sys
import time
import textwrap
from pathlib import Path
from datetime import datetime

# ── Make sure project root is on sys.path regardless of CWD ──────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def banner(title: str, width: int = 66):
    print("\n" + "+" + "=" * width + "+")
    print("|" + title.center(width) + "|")
    print("+" + "=" * width + "+")


def section(title: str):
    print(f"\n{'-' * 66}")
    print(f"  >> {title}")
    print(f"{'-' * 66}")


# ─────────────────────────────────────────────────────────────────────────────
def run():
    start_time = time.time()
    banner("Snap Inc. (Snapchat)  |  AI Deployment Strategy Automation")
    print(f"  Run started : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    print(f"  Project root: {PROJECT_ROOT}")

    # ── Step 1: Data Generation ───────────────────────────────────────────
    section("STEP 1 — Generating Synthetic Datasets")
    from data.data_generator import generate_all
    datasets = generate_all(save=True)

    # ── Step 2: AR Lens Model ─────────────────────────────────────────────
    section("STEP 2 — Training AR Lens Recommendation Model")
    from models.ar_lens_recommender import train_ar_model
    ar_metrics = train_ar_model(datasets["ar_lens"])

    # ── Step 3: Ad Targeting Model ────────────────────────────────────────
    section("STEP 3 — Training Programmatic Ad Targeting Model")
    from models.ad_targeting_model import train_ad_model
    ad_metrics = train_ad_model(datasets["ad_targeting"])

    # ── Step 4: Churn Prediction Model ────────────────────────────────────
    section("STEP 4 — Training User Churn Prediction Model")
    from models.churn_predictor import train_churn_model
    churn_metrics = train_churn_model(datasets["churn"])

    # ── Step 5: ROI Analysis ──────────────────────────────────────────────
    section("STEP 5 — Running ROI Analysis")
    from roi.roi_calculator import run_roi_analysis
    roi_summary = run_roi_analysis(ar_metrics, ad_metrics, churn_metrics)

    # ── Step 6: Text Report ───────────────────────────────────────────────
    section("STEP 6 — Writing Consolidated Report")
    write_report(ar_metrics, ad_metrics, churn_metrics, roi_summary)

    elapsed = time.time() - start_time
    banner(f"Pipeline Complete  |  Elapsed: {elapsed:.1f}s")
    print(f"  All outputs saved to: {REPORTS_DIR}\n")


# ─────────────────────────────────────────────────────────────────────────────
def write_report(ar: dict, ad: dict, churn: dict, roi_df):
    """Write a structured plain-text executive report."""

    lines = []

    def h1(t):   lines.append("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)
    def h2(t):   lines.append(f"\n{'-' * 60}\n  {t}\n{'-' * 60}")
    def para(t): lines.extend(textwrap.wrap(t, width=68, initial_indent="  ", subsequent_indent="  ")); lines.append("")
    def kv(k, v): lines.append(f"  {k:<35} {v}")

    h1("Snap Inc. (Snapchat)  —  AI Deployment Strategy Report")
    lines.append(f"  Generated : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
    lines.append(f"  Course    : MSc Data Science  |  AI Business Case Study\n")

    # ── Executive Summary ─────────────────────────────────────────────────
    h2("1. Executive Summary")
    para(
        "This report documents the design and automated evaluation of three AI-driven "
        "initiatives for Snap Inc. (Snapchat). The use-cases span product (AR Lens "
        "Recommendation), monetisation (Programmatic Ad Targeting), and retention "
        "(User Churn Prediction). Each initiative is backed by a machine-learning model "
        "trained on synthetic yet realistic data, and quantified through an ROI framework "
        "grounded in publicly disclosed Snap business metrics."
    )

    # ── Company & Industry Context ────────────────────────────────────────
    h2("2. Company & Industry Context")
    para(
        "Snap Inc. operates in the Social Media & Digital Advertising industry. "
        "As of Q4 2024, Snapchat reported 422M DAU, \$5.4B annual revenue, and "
        "a core product centred on AR-powered ephemeral messaging. "
        "AI is central to Snap's strategy across: (a) generative AR lens creation, "
        "(b) Snap AI chatbot, (c) personalised content ranking, and "
        "(d) advertiser targeting. The three use-cases below represent the highest-ROI "
        "opportunities across product and monetisation."
    )

    # ── Data Strategy ─────────────────────────────────────────────────────
    h2("3. Data Strategy")
    para("Three distinct datasets were synthesised to mirror Snapchat's real-world data assets:")
    lines.append("  +-----------------------------------------------------------------+")
    lines.append("  |  Dataset            |  Rows   |  Features |  Target             |")
    lines.append("  +-----------------------------------------------------------------+")
    lines.append("  |  AR Lens            |  5,000  |  13       |  AR engagement (bin)|")
    lines.append("  |  Ad Targeting       |  8,000  |  12       |  Click-through (bin)|")
    lines.append("  |  User Churn         |  6,000  |  15       |  Churn (binary)     |")
    lines.append("  +-----------------------------------------------------------------+")
    lines.append("")
    para(
        "Key data sources in a production deployment would include: Snapchat app event "
        "logs (Kafka streams), Ads Manager auction data, CRM/push notification logs, "
        "and third-party panel data. All data processing would comply with GDPR, CCPA, "
        "and Snap's Privacy Principles, with differential privacy applied where required."
    )

    # ── Model Choices ─────────────────────────────────────────────────────
    h2("4. Model Selection & Performance")

    h2("  4a. AR Lens Recommendation  —  Random Forest")
    para("Rationale: Random Forest provides interpretable feature importances essential "
         "for product decisions (e.g. which lens attributes drive engagement). It "
         "handles mixed feature types natively and is robust to outliers.")
    kv("Algorithm:",              "Random Forest Classifier")
    kv("n_estimators:",           "200")
    kv("Max depth:",              "10  (prevents overfitting)")
    kv("Class weighting:",        "Balanced")
    kv("CV ROC-AUC (5-fold):",    f"{ar['cv_roc_auc']:.4f}")
    kv("Test ROC-AUC:",           f"{ar['test_roc_auc']:.4f}")
    kv("Test Precision (class 1):", f"{ar['classification_report']['1']['precision']:.4f}")
    kv("Test Recall    (class 1):", f"{ar['classification_report']['1']['recall']:.4f}")
    kv("Test F1        (class 1):", f"{ar['classification_report']['1']['f1-score']:.4f}")
    lines.append("")

    h2("  4b. Programmatic Ad Targeting  —  XGBoost")
    para("Rationale: XGBoost is the industry standard for click-through rate modelling "
         "due to its ability to capture nonlinear interactions between user segments, "
         "ad formats, and contextual signals (time-of-day, placement). It scales to "
         "billions of impressions via online learning extensions.")
    kv("Algorithm:",              "XGBoost Classifier")
    kv("n_estimators:",           "300")
    kv("Learning rate:",          "0.05  (shrinkage)")
    kv("Imbalance handling:",     "scale_pos_weight")
    kv("CV ROC-AUC (5-fold):",    f"{ad['cv_roc_auc']:.4f}")
    kv("Test ROC-AUC:",           f"{ad['test_roc_auc']:.4f}")
    kv("Average Precision:",      f"{ad['avg_precision']:.4f}")
    kv("Test Precision (class 1):", f"{ad['classification_report']['1']['precision']:.4f}")
    kv("Test Recall    (class 1):", f"{ad['classification_report']['1']['recall']:.4f}")
    lines.append("")

    h2("  4c. User Churn Prediction  —  XGBoost + SMOTE")
    para("Rationale: Churn datasets are inherently imbalanced (most users are retained). "
         "SMOTE generates synthetic minority samples to improve recall for at-risk users, "
         "which is the critical business metric. Threshold tuning further optimises "
         "the F1 trade-off between precision and recall.")
    kv("Algorithm:",              "XGBoost Classifier + SMOTE")
    kv("Imbalance handling:",     "SMOTE (k=5 neighbours)")
    kv("Threshold tuning:",       "F1-optimised sweep [0.1, 0.9]")
    kv("Optimal threshold:",      f"{churn['optimal_threshold']}")
    kv("CV ROC-AUC (5-fold):",    f"{churn['cv_roc_auc']:.4f}")
    kv("Test ROC-AUC:",           f"{churn['test_roc_auc']:.4f}")
    kv("Test Precision (class 1):", f"{churn['classification_report']['1']['precision']:.4f}")
    kv("Test Recall    (class 1):", f"{churn['classification_report']['1']['recall']:.4f}")
    kv("Test F1        (class 1):", f"{churn['classification_report']['1']['f1-score']:.4f}")
    lines.append("")

    # ── ROI Metrics ───────────────────────────────────────────────────────
    h2("5. ROI Metrics")
    para(
        "ROI estimates are derived from publicly available Snap financials (Q4 2024 "
        "10-Q) and industry benchmarks. Revenue uplifts are conservatively modelled "
        "as a function of model ROC-AUC relative to a random baseline."
    )

    for _, row in roi_df.iterrows():
        lines.append(f"\n  >> {row['Use Case']}")
        kv("  Annual Revenue Gain:", f"${row['Annual Revenue Gain ($)']:,.0f}")
        kv("  Annual Cost:",         f"${row['Annual Cost ($)']:,.0f}")
        kv("  Net Benefit:",         f"${row['Net Benefit ($)']:,.0f}")
        kv("  ROI:",                 f"{row['ROI (%)']:.1f}%")

    lines.append("")
    total_net    = roi_df["Net Benefit ($)"].sum()
    total_invest = roi_df["Annual Cost ($)"].sum()
    blended_roi  = (total_net - total_invest) / total_invest * 100
    lines.append("  --- Consolidated -------------------------------------------")
    kv("  Total Annual Investment:",  f"${total_invest/1e6:.2f}M")
    kv("  Total Annual Net Benefit:", f"${total_net/1e6:.1f}M")
    kv("  Blended ROI:",              f"{blended_roi:.0f}%")
    lines.append("")

    # ── Deployment Architecture ───────────────────────────────────────────
    h2("6. Deployment Architecture")
    para(
        "Production deployment would follow a MLOps pipeline: "
        "(1) Feature Store (Feast/Tecton) for real-time feature serving, "
        "(2) Model Registry (MLflow) for versioning and A/B gating, "
        "(3) Serving layer via FastAPI + Kubernetes for low-latency inference, "
        "(4) Shadow mode deployment before full rollout, "
        "(5) Continuous monitoring with Evidently AI for data drift, "
        "(6) Retraining triggers on KPI degradation > 5% over 7-day window."
    )

    # ── Ethical & Compliance Considerations ──────────────────────────────
    h2("7. Ethical & Compliance Considerations")
    para(
        "All three models are designed with fairness in mind: "
        "(a) Demographic features are excluded from ad targeting to prevent "
        "discriminatory ad delivery (per EU AI Act and UK ICO guidelines). "
        "(b) Churn model outputs are gated through human review before any "
        "win-back communication is sent to minors (13-17 age group). "
        "(c) GDPR Article 22 compliance: users can opt out of automated decision-making. "
        "(d) Model explainability reports (SHAP values) are generated quarterly "
        "for internal audit."
    )

    # ── Conclusion ────────────────────────────────────────────────────────
    h2("8. Conclusion")
    para(
        "The three AI initiatives collectively deliver an estimated blended ROI "
        f"of {blended_roi:.0f}%, with a total annual net benefit of "
        f"\${total_net/1e6:.1f}M against an investment of \${total_invest/1e6:.2f}M. "
        "The highest individual ROI is from Ad Targeting, which leverages Snap's "
        "existing impression scale. Churn Prediction delivers the strongest "
        "strategic value by protecting DAU — the metric most scrutinised by investors. "
        "AR Lens Recommendation directly supports Snap's core product differentiation."
    )

    h1("END OF REPORT")

    report_text = "\n".join(lines)
    report_path = REPORTS_DIR / "snap_ai_strategy_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\n  [Report saved] {report_path}")
    print(report_text)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run()

