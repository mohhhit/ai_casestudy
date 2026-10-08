"""
================================================================================
 Snap Inc. (Snapchat) — AI Deployment Strategy
 Module: roi/roi_calculator.py
 Purpose: Calculate and visualise Return on Investment (ROI) metrics
          for all three AI initiatives based on model performance and
          realistic Snapchat business assumptions.
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Business Constants  (based on publicly available Snap financials & estimates)
# ---------------------------------------------------------------------------

SNAP_DAU              = 422_000_000   # Daily Active Users (Q4 2024 estimate)
SNAP_ARPU_USD         = 3.10          # Average Revenue Per User per quarter (USD)
AD_REVENUE_PER_Q      = 1_360_000_000 # ~$1.36B quarterly ad revenue
AR_LENS_DAILY_USES    = 250_000_000   # AR lens daily interactions (Snap stat)
COST_PER_MODEL_DEV    = 800_000       # Estimated annual ML model dev cost per use-case
INFRA_COST_PER_YEAR   = 500_000       # Cloud infra / serving per use-case

# ---------------------------------------------------------------------------
# 1.  AR Lens Recommendation ROI
# ---------------------------------------------------------------------------

def calculate_ar_roi(test_roc_auc: float, baseline_engagement_rate: float = 0.35) -> dict:
    """
    Estimate ROI for AI-driven AR Lens Recommendation.

    Business logic
    --------------
    - Baseline: 35 % of AR lens impressions lead to user interaction
    - Model lifts engagement by modelling-derived uplift (scaled from AUC)
    - Each additional engagement generates ad-equivalent impression value
    """
    # Uplift: linearly scale AUC above random (0.5) to ~15% max lift
    model_uplift    = max(0, (test_roc_auc - 0.50) / 0.50) * 0.15
    new_engage_rate = min(baseline_engagement_rate + model_uplift, 0.95)

    daily_extra_engagements = AR_LENS_DAILY_USES * (new_engage_rate - baseline_engagement_rate)
    # Each AR engagement ~ $0.003 CPM equivalent (brand recall, re-engagement)
    value_per_engage_usd    = 0.003
    annual_revenue_gain     = daily_extra_engagements * value_per_engage_usd * 365

    total_cost = (COST_PER_MODEL_DEV + INFRA_COST_PER_YEAR)
    roi_pct    = (annual_revenue_gain - total_cost) / total_cost * 100

    return {
        "use_case":                "AR Lens Recommendation",
        "model":                   "Random Forest",
        "roc_auc":                 test_roc_auc,
        "baseline_engagement":     f"{baseline_engagement_rate:.0%}",
        "model_engagement":        f"{new_engage_rate:.2%}",
        "engagement_lift":         f"{model_uplift:.2%}",
        "annual_extra_engagements": int(daily_extra_engagements * 365),
        "annual_revenue_gain_usd": round(annual_revenue_gain, 2),
        "annual_cost_usd":         total_cost,
        "net_benefit_usd":         round(annual_revenue_gain - total_cost, 2),
        "roi_pct":                 round(roi_pct, 1),
    }


# ---------------------------------------------------------------------------
# 2.  Ad Targeting ROI
# ---------------------------------------------------------------------------

def calculate_ad_roi(test_roc_auc: float, baseline_ctr: float = 0.048) -> dict:
    """
    Estimate ROI for AI-driven Programmatic Ad Targeting.

    Business logic
    --------------
    - Baseline CTR: 4.8 % (Snap industry average)
    - Better targeting drives higher effective CPM → revenue increase
    - eCPM baseline ~$3.00; each 10% CTR lift ~ $0.30 eCPM improvement
    """
    ctr_lift_factor = max(0, (test_roc_auc - 0.50) / 0.50) * 0.25
    new_ctr         = baseline_ctr * (1 + ctr_lift_factor)

    baseline_ecpm = 3.00
    ecpm_gain     = baseline_ecpm * ctr_lift_factor * 0.5   # conservative 50% pass-through
    new_ecpm      = baseline_ecpm + ecpm_gain

    # Impressions per quarter estimate (from ad revenue / ecpm)
    quarterly_impressions = (AD_REVENUE_PER_Q / (baseline_ecpm / 1000))
    revenue_gain_q        = quarterly_impressions * (new_ecpm - baseline_ecpm) / 1000
    annual_revenue_gain   = revenue_gain_q * 4

    total_cost = (COST_PER_MODEL_DEV + INFRA_COST_PER_YEAR)
    roi_pct    = (annual_revenue_gain - total_cost) / total_cost * 100

    return {
        "use_case":               "Programmatic Ad Targeting",
        "model":                  "XGBoost",
        "roc_auc":                test_roc_auc,
        "baseline_ctr":           f"{baseline_ctr:.2%}",
        "model_ctr":              f"{new_ctr:.2%}",
        "ctr_lift":               f"{ctr_lift_factor:.2%}",
        "baseline_ecpm_usd":      baseline_ecpm,
        "model_ecpm_usd":         round(new_ecpm, 4),
        "annual_revenue_gain_usd": round(annual_revenue_gain, 2),
        "annual_cost_usd":        total_cost,
        "net_benefit_usd":        round(annual_revenue_gain - total_cost, 2),
        "roi_pct":                round(roi_pct, 1),
    }


# ---------------------------------------------------------------------------
# 3.  Churn Prediction ROI
# ---------------------------------------------------------------------------

def calculate_churn_roi(test_roc_auc: float, recall_churned: float,
                         baseline_churn_rate: float = 0.12) -> dict:
    """
    Estimate ROI for AI-driven User Churn Prediction.

    Business logic
    --------------
    - Baseline churn rate: 12% of DAU churn per quarter
    - Model identifies at-risk users → win-back campaign (email/push)
    - Win-back success rate: 20% of correctly identified churners
    - Each retained user worth ~1 quarter ARPU
    """
    at_risk_users         = SNAP_DAU * baseline_churn_rate
    correctly_identified  = at_risk_users * recall_churned
    win_back_rate         = 0.20
    users_retained        = correctly_identified * win_back_rate

    revenue_retained      = users_retained * SNAP_ARPU_USD    # per quarter
    annual_revenue_gain   = revenue_retained * 4               # annualised

    # Win-back campaign cost (~$0.10 per notification/email sent)
    campaign_cost         = correctly_identified * 0.10 * 4
    total_cost            = COST_PER_MODEL_DEV + INFRA_COST_PER_YEAR + campaign_cost
    roi_pct               = (annual_revenue_gain - total_cost) / total_cost * 100

    return {
        "use_case":               "User Churn Prediction",
        "model":                  "XGBoost + SMOTE",
        "roc_auc":                test_roc_auc,
        "recall_churned":         f"{recall_churned:.2%}",
        "baseline_churn_rate":    f"{baseline_churn_rate:.0%}",
        "at_risk_users_per_q":    int(at_risk_users),
        "users_retained_per_q":   int(users_retained),
        "annual_revenue_gain_usd": round(annual_revenue_gain, 2),
        "annual_cost_usd":        round(total_cost, 2),
        "net_benefit_usd":        round(annual_revenue_gain - total_cost, 2),
        "roi_pct":                round(roi_pct, 1),
    }


# ---------------------------------------------------------------------------
# Aggregated Summary & Visualisation
# ---------------------------------------------------------------------------

def compile_roi_summary(ar_results: dict, ad_results: dict, churn_results: dict) -> pd.DataFrame:
    """Build a tidy summary DataFrame from the three ROI dicts."""
    rows = []
    for r in [ar_results, ad_results, churn_results]:
        rows.append({
            "Use Case":               r["use_case"],
            "Model":                  r["model"],
            "ROC-AUC":                r["roc_auc"],
            "Annual Revenue Gain ($)": r["annual_revenue_gain_usd"],
            "Annual Cost ($)":        r["annual_cost_usd"],
            "Net Benefit ($)":        r["net_benefit_usd"],
            "ROI (%)":                r["roi_pct"],
        })
    return pd.DataFrame(rows)


def plot_roi_dashboard(summary_df: pd.DataFrame,
                       ar: dict, ad: dict, ch: dict):
    """
    Generate a 4-panel ROI dashboard:
      Panel 1: Net Benefit bar chart
      Panel 2: ROI % comparison
      Panel 3: Cost vs Revenue waterfall
      Panel 4: Snap KPI impact table
    """
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Snap Inc.  |  AI Deployment Strategy - ROI Dashboard",
                 fontsize=16, fontweight="bold", y=0.98)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    use_cases = summary_df["Use Case"].tolist()
    colors    = ["#FFFC00", "#FF6900", "#8B5CF6"]   # Snapchat-ish palette

    # --- Panel 1: Net Benefit
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(use_cases, summary_df["Net Benefit ($)"] / 1e6, color=colors, edgecolor="black", linewidth=0.8)
    ax1.set_title("Estimated Annual Net Benefit", fontsize=12, fontweight="bold")
    ax1.set_ylabel("USD (Millions)")
    ax1.set_xticks(range(len(use_cases)))
    ax1.set_xticklabels([uc.replace(" ", "\n") for uc in use_cases], fontsize=9)
    for bar, val in zip(bars, summary_df["Net Benefit ($)"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"${val/1e6:.1f}M", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- Panel 2: ROI %
    ax2 = fig.add_subplot(gs[0, 1])
    roi_vals = summary_df["ROI (%)"].tolist()
    hbars = ax2.barh(use_cases, roi_vals, color=colors, edgecolor="black", linewidth=0.8)
    ax2.set_title("ROI % per Use-Case", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Return on Investment (%)")
    ax2.axvline(0, color="black", linewidth=0.8)
    for bar, val in zip(hbars, roi_vals):
        ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f}%", va="center", fontsize=10, fontweight="bold")

    # --- Panel 3: Cost vs Revenue stacked comparison
    ax3 = fig.add_subplot(gs[1, 0])
    x    = np.arange(len(use_cases))
    cost = summary_df["Annual Cost ($)"].values / 1e6
    rev  = summary_df["Annual Revenue Gain ($)"].values / 1e6
    width = 0.35
    ax3.bar(x - width/2, cost, width, label="Annual Cost",    color="#EF4444", alpha=0.85)
    ax3.bar(x + width/2, rev,  width, label="Revenue Gain",   color="#22C55E", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels([uc.replace(" ", "\n") for uc in use_cases], fontsize=9)
    ax3.set_ylabel("USD (Millions)")
    ax3.set_title("Cost vs Revenue Gain", fontsize=12, fontweight="bold")
    ax3.legend()

    # --- Panel 4: KPI Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    kpi_data = [
        ["Metric",                         "Value"],
        ["Snap DAU",                        f"{SNAP_DAU/1e6:.0f}M"],
        ["Quarterly Ad Revenue",            f"${AD_REVENUE_PER_Q/1e9:.2f}B"],
        ["ARPU (quarterly)",                f"${SNAP_ARPU_USD}"],
        ["AR Lens Daily Interactions",      f"{AR_LENS_DAILY_USES/1e6:.0f}M"],
        ["Total Net Annual Benefit",        f"${summary_df['Net Benefit ($)'].sum()/1e6:.1f}M"],
        ["Total Annual Investment",         f"${summary_df['Annual Cost ($)'].sum()/1e6:.2f}M"],
        ["Blended ROI",                     f"{summary_df['ROI (%)'].mean():.0f}%"],
    ]
    table = ax4.table(
        cellText=kpi_data[1:],
        colLabels=kpi_data[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.4, 1.8)
    # Style header row
    for j in range(2):
        table[(0, j)].set_facecolor("#1C1C1E")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax4.set_title("Snap Inc. - Business KPIs", fontsize=12, fontweight="bold")

    out_path = REPORTS_DIR / "roi_dashboard.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  [Plot saved] {out_path}")


def run_roi_analysis(ar_metrics: dict, ad_metrics: dict, churn_metrics: dict) -> pd.DataFrame:
    """
    Orchestrate full ROI analysis using model metrics.

    Parameters
    ----------
    ar_metrics    : Output dict from train_ar_model()
    ad_metrics    : Output dict from train_ad_model()
    churn_metrics : Output dict from train_churn_model()
    """
    print("\n" + "=" * 60)
    print("  ROI ANALYSIS -- Snap Inc. AI Deployment Strategy")
    print("=" * 60)

    recall_churned = churn_metrics["classification_report"]["1"]["recall"]

    ar_roi    = calculate_ar_roi(ar_metrics["test_roc_auc"])
    ad_roi    = calculate_ad_roi(ad_metrics["test_roc_auc"])
    churn_roi = calculate_churn_roi(churn_metrics["test_roc_auc"], recall_churned)

    summary = compile_roi_summary(ar_roi, ad_roi, churn_roi)

    print("\n" + summary.to_string(index=False))

    total_net    = summary["Net Benefit ($)"].sum()
    total_invest = summary["Annual Cost ($)"].sum()
    blended_roi  = (total_net - total_invest) / total_invest * 100

    print(f"\n  -- Consolidated ------------------------------------------")
    print(f"  Total Annual Investment : ${total_invest/1e6:.2f}M")
    print(f"  Total Net Annual Benefit: ${total_net/1e6:.1f}M")
    print(f"  Blended ROI             : {blended_roi:.0f}%")

    plot_roi_dashboard(summary, ar_roi, ad_roi, churn_roi)

    # Save summary CSV
    csv_path = REPORTS_DIR / "roi_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  [CSV saved]  {csv_path}")

    return summary
