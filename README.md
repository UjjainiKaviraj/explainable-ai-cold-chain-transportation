# Explainable AI for U.S. Cold-Chain Transportation
### Predicting Refrigerated Truck Rates, Capacity, and Financial Impact

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-2980B9)
![SHAP](https://img.shields.io/badge/SHAP-explainability-brightgreen)
![Status](https://img.shields.io/badge/status-complete-success)

---

## Overview

Logistics planners in U.S. refrigerated trucking face a fundamental problem: the USDA's market reports
are backward-looking. They tell you what happened last quarter — not what's about to happen next.
When conditions shift unfavorably, planners scramble to book trucks at spot-market rates that can be
2–3x contracted rates, directly eroding margins on perishable supply chains.

This project closes that gap. Drawing on **25 years of USDA Agricultural Marketing Service data (2000–2025)**,
we built a two-component machine learning system:

- A **classification model** that predicts weekly truck availability (surplus / balanced / shortage)
- A **regression model** that forecasts quarterly refrigerated truck rates per mile
- **SHAP-based explainability** that translates model outputs into actionable, interpretable signals
- A **financial impact module** that converts predictions into dollar figures planners can act on

> **Course:** DSCI 5260 — Business Capstone, University of North Texas | **Group 5** | May 2026

---

## Results at a Glance

| Task | Model | Metric | Target | Achieved |
|------|-------|--------|--------|----------|
| Truck Availability Classification | Random Forest (Tuned) | Weighted F1 | ≥ 0.65 recall | **0.850** |
| Truck Availability Classification | Random Forest (Tuned) | Shortage Recall | ≥ 65% | **70.5%** |
| Rate Forecasting | LightGBM | MAE | ≤ $0.50/mile | **$0.29/mile** |
| Rate Forecasting | LightGBM | MAPE | — | **4.50%** |
| Rate Forecasting | LightGBM | R² | ≥ 0.80 | **0.851** |
| Baseline (Seasonal Naïve) | — | MAE | — | $1.96/mile |

LightGBM's rate forecasts are **6.7× more accurate** than the seasonal naïve baseline.

---

## Key Findings

**Classification (Truck Availability)**
- Random Forest (Tuned) achieved Weighted F1 = 0.850 and Shortage Recall = 0.705 on the chronological holdout. The model correctly identifies 7 out of 10 shortage weeks before they affect operations.
- `Lag_Availability` (SHAP = 0.691) is the single dominant predictor — market conditions are sticky. Shortages persist week-to-week, making the prior week's rating the strongest signal of the next.
- SMOTE oversampling was tested and *rejected*: QWK degraded from 0.680 to 0.419 because synthetic minority samples cannot replicate real shortage dynamics in time-ordered data.

**Regression (Rate Forecasting)**
- LightGBM achieved MAE = $0.29/mile and MAPE = 4.50% — meaning forecasts land within 4.5% of actual rates, on average.
- `Lag_Rate`, `US_Avg_Rate`, and `Rate_Gap_vs_US` together explain approximately 74% of rate predictions.
- `Peak_Season` (Q1/Q4) and `Spike_Period` (2021–22) are the next most important SHAP drivers, confirming that seasonal patterns and structural breaks are the two risk regimes planners most need to anticipate.

**Financial Impact**
- A single high-cost quarter during the 2021–22 spike generated an estimated **$509 million** in transportation cost overruns across the market.
- Q1 carries the highest revenue-at-risk of any quarter. A company spending $10M/quarter on reefer freight faces ~$450K in planning variance per quarter at 4.5% MAPE.

---

## Decision Rule

> When **predicted rate > $4.00/mile** AND **shortage probability > 35%** → flag as **HIGH URGENCY**
>
> Action: initiate early contracting 4–6 weeks ahead of the quarter.

This rule is derived from threshold sensitivity analysis in Section 8 of the modeling notebook. The $4.00/mile threshold corresponds to the post-structural-break floor in the market.

---

## Repository Structure

```
explainable-ai-cold-chain-transportation/
│
├── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb                         # Exploratory data analysis across all 5 datasets
│   └── 02_modeling_and_explainability.ipynb # Full ML pipeline: features → models → SHAP → financial impact
│
├── data/
│   ├── raw/                                 # Original USDA AMS Excel files (5 datasets)
│   ├── processed/
│   │   ├── classification_dataset.csv       # Feature-engineered, model-ready (n=32,556)
│   │   └── regression_dataset.csv           # Feature-engineered, model-ready (n=4,428)
│   └── README.md                            # Data dictionary + source documentation
│
├── reports/
│   ├── Group-5_Final_Project_Report.pdf     # Full 32-page report (CRISP-DM)
│   └── ColdChain_Group5_Final_Presentation.pptx
│
├── assets/                                  # Charts embedded in this README
│
├── requirements.txt
└── .gitignore
```

---

## Notebooks

### `01_EDA.ipynb`
Covers all five USDA AMS datasets. Key outputs:
- Class imbalance analysis (shortage = 11.3% of records — establishing why accuracy is a misleading metric)
- 25-year national rate trend, including the 2021–22 structural break that saw rates nearly double to $5.41/mile
- Geographic concentration: Mexico crossings + California drive 70%+ of refrigerated shipment volume
- Commodity distribution: potatoes, citrus, and sweet potatoes dominate

### `02_modeling_and_explainability.ipynb`
Nine sections covering the full pipeline:

| Section | Content |
|---------|---------|
| 0 | Setup and package installation |
| 1 | Data loading and inspection |
| 2 | Feature engineering and chronological train/test split |
| 3 | Classification models (9 models benchmarked) |
| 4 | SHAP explainability — classification |
| 5 | Regression models (6 models benchmarked, including Prophet) |
| 6 | SHAP explainability — regression |
| 7 | Financial impact analysis |
| 8 | Decision support rule engine |
| 9 | Final model summary |

> **Note on methodology:** All cross-validation used `TimeSeriesSplit` (5 folds). Random k-fold was
> explicitly ruled out after early experiments confirmed it inflated performance by 10–15 percentage
> points by allowing future data to leak into training folds.

---

## Data

All data is sourced from the **USDA Agricultural Marketing Service Agricultural Refrigerated Truck Quarterly Datasets**, publicly available at [ams.usda.gov](https://www.ams.usda.gov).

| Dataset | Granularity | Rows | Role in Project |
|---------|------------|------|-----------------|
| Weekly Truck Availability by Origin & Commodity | Weekly | 32,556 | Classification target |
| Quarterly Rates by Origin & Distance | Quarterly | 4,428 | Regression target |
| Quarterly Rates by O-D Pair | Quarterly | 8,051 | Lane-level features |
| U.S. Average Rates by Distance | Quarterly | 404 | National benchmark & imputation |
| Shipment Volumes by Origin & Commodity | Quarterly | 71,617 | Demand-side features |

**Data quality note:** The rates dataset has ~30% missing values. These were imputed using the U.S. average rate for the same quarter and distance band. An `imputation_flag` and `rate_gap_vs_us` feature were engineered to preserve transparency and allow the model to learn from missingness patterns.

---

## Methodology

The project followed the **CRISP-DM framework** across six phases: Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, and Deployment Planning.

**Feature engineering highlights:**
- Lag features (`Lag_Availability`, `Lag_Rate`) to capture week-to-week and quarter-to-quarter persistence
- Cyclical temporal encodings (`year_sin`, `year_cos`) so Q4 and Q1 are treated as adjacent rather than distant
- `Spike_Period` binary flag for 2021–22 structural break
- `Volume_Season_Interaction`: national shipment tons × peak season flag
- `Rate_Gap_vs_US`: lane rate deviation from national benchmark

**Class imbalance:** Balanced class weights were applied (shortage class weight = 2.70). SMOTE was tested and rejected.

---

## Recommendations for Practitioners

1. **Pre-book capacity 4–6 weeks ahead in Q1 and Q4.** SHAP confirms `Peak_Season` is a top-5 driver of both shortage and rate spikes.
2. **Monitor Mexico crossings and California first.** These origins drive 70%+ of shipment volume and are where market pressure concentrates.
3. **Apply the HIGH URGENCY decision rule** (rate > $4.00/mile AND shortage probability > 35%) as the trigger for early contracting.
4. **Retrain models on a rolling 12-quarter window** each time new USDA data is published to account for structural drift.

---

## Setup

```bash
git clone https://github.com/<your-username>/explainable-ai-cold-chain-transportation.git
cd explainable-ai-cold-chain-transportation
pip install -r requirements.txt
```

Open the notebooks in order: `01_EDA.ipynb` → `02_modeling_and_explainability.ipynb`.

**Google Colab users:** The modeling notebook was originally developed in Colab. Update the data paths in Section 1 from `/content/` to your local `data/processed/` directory.

---

## Requirements

```
lightgbm
shap
imbalanced-learn
prophet
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

---

## Team

**Group 5 — University of North Texas, DSCI 5260**

Yeswanthi Gadiraju · Ujjaini Kaviraj · Bhanu Prakash Chippada · Anushka Singh · Saniya Naffisa Shaik

*Data provided by the U.S. Department of Agriculture Agricultural Marketing Service.*

---

## License

This project is for academic and portfolio purposes. The underlying USDA AMS data is publicly available under standard U.S. government open data terms.
