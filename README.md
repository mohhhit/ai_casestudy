# 🟡 Snap Inc. — AI Deployment Strategy
### MSc Data Science | AI Business Case Study

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2-green)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Overview

This project automates a complete **AI deployment strategy** for **Snap Inc. (Snapchat)** across three high-value business use-cases:

| # | Use-Case | Model | Goal |
|---|----------|-------|------|
| 1 | **AR Lens Recommendation** | Random Forest | Increase AR engagement rate |
| 2 | **Programmatic Ad Targeting** | XGBoost (CTR) | Improve click-through & eCPM |
| 3 | **User Churn Prediction** | XGBoost + SMOTE | Retain at-risk DAU |

Each initiative is backed by:
- Synthetic-but-realistic datasets mirroring Snapchat's real data assets
- Trained & cross-validated ML models with full diagnostic plots
- An ROI framework grounded in publicly disclosed Snap financials (Q4 2024)
- A consolidated 8-section executive report

---

## 🗂️ Project Structure

```
ai_casestudy/
├── data/
│   ├── data_generator.py       # Synthetic dataset generator
│   ├── ar_lens_data.csv        # AR engagement dataset (5,000 rows)
│   ├── ad_targeting_data.csv   # Ad impression dataset  (8,000 rows)
│   └── churn_data.csv          # User churn dataset     (6,000 rows)
├── models/
│   ├── ar_lens_recommender.py  # Random Forest model
│   ├── ad_targeting_model.py   # XGBoost CTR model
│   └── churn_predictor.py      # XGBoost + SMOTE churn model
├── roi/
│   └── roi_calculator.py       # ROI computation & dashboard
├── pipeline/
│   └── run_pipeline.py         # End-to-end orchestrator
├── reports/                    # Auto-generated outputs (plots, CSV, report)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/mohhhit/ai_casestudy.git
cd ai_casestudy
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline
```bash
# On Windows (PowerShell)
$env:PYTHONUTF8="1"; python -m pipeline.run_pipeline

# On Linux / macOS
PYTHONUTF8=1 python -m pipeline.run_pipeline
```

---

## 📊 Results

### Model Performance

| Model | Use-Case | Test ROC-AUC |
|-------|----------|-------------|
| Random Forest | AR Lens Recommendation | 0.596 |
| XGBoost | Ad Targeting (CTR) | 0.562 |
| XGBoost + SMOTE | Churn Prediction | 0.645 |

### ROI Summary (Annual)

| Initiative | Revenue Gain | Cost | Net Benefit | ROI |
|------------|-------------|------|-------------|-----|
| AR Lens Recommendation | $7.85M | $1.3M | $6.55M | 504% |
| Programmatic Ad Targeting | $84.3M | $1.3M | $83.0M | 6,384% |
| User Churn Prediction | $29.7M | $6.1M | $23.6M | 388% |
| **Consolidated** | **$121.8M** | **$8.69M** | **$113.1M** | **1,202%** |

> ROI is conservatively modelled as a linear function of model AUC relative to a random baseline, using Snap Q4 2024 financial disclosures.

---

## 📁 Generated Reports

After running the pipeline, the `reports/` directory contains:

| File | Description |
|------|-------------|
| `ar_lens_model_results.png` | Confusion matrix + ROC curve + feature importance |
| `ad_targeting_model_results.png` | Confusion matrix + PR curve + XGBoost feature gain |
| `churn_model_results.png` | Threshold-tuned CM + ROC curve + feature importance |
| `roi_dashboard.png` | 4-panel ROI dashboard with Snap KPI table |
| `roi_summary.csv` | ROI figures in tabular form |
| `snap_ai_strategy_report.txt` | Full 8-section executive report |

---

## 🧠 Technical Details

### Data Strategy
- All datasets are **synthetically generated** with probabilistically grounded feature interactions to mimic real Snapchat telemetry
- In production, data would flow from: Kafka event streams → Feature Store (Feast/Tecton) → Model serving layer

### ML Pipeline
- **Imbalance handling**: `scale_pos_weight` (XGBoost) and SMOTE oversampling
- **Validation**: 5-fold Stratified K-Fold cross-validation
- **Threshold tuning**: F1-optimised sweep over [0.10, 0.90] for churn model
- **MLOps**: Production deployment via MLflow + FastAPI + Kubernetes (architectural recommendation)

### Ethics & Compliance
- Demographic features excluded from ad targeting (EU AI Act, UK ICO)
- GDPR Article 22 opt-out for automated decisions
- Churn outputs for minors (13–17) gated through human review
- Quarterly SHAP explainability audits

---

## 📦 Dependencies

```
numpy >= 1.24
pandas >= 2.0
scikit-learn >= 1.3
xgboost >= 1.7
imbalanced-learn >= 0.11
matplotlib >= 3.7
seaborn >= 0.12
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Submitted as part of MSc Data Science — AI Business Case Study Practical*
