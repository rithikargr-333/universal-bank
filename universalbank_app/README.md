# 🏦 Universal Bank — Loan Intelligence Hub

A full-stack **Streamlit analytics dashboard** for predicting personal loan acceptance and designing hyper-personalised marketing campaigns.

## 🚀 Live Demo
Deploy instantly on [Streamlit Community Cloud](https://streamlit.io/cloud) — free hosting, no setup needed.

## 📦 Features

| Section | What you get |
|---|---|
| **📊 Executive Overview** | KPI cards, acceptance rate donut, demographic breakdown |
| **🔍 Descriptive Analytics** | Age, education, family, income, CC spend, mortgage, bank product distributions |
| **📈 Diagnostic Analytics** | Correlation heatmap, violin plots, income×family acceptance heatmap |
| **🤖 Predictive Models** | Decision Tree, Random Forest, Gradient Boosted Tree — accuracy, precision, recall, F1, AUC table; single ROC curve; labelled confusion matrices with % values; feature importance |
| **🎯 Prescriptive Analytics** | 4-tier campaign segments (Platinum/Gold/Silver/General), campaign playbook, budget allocation chart |
| **📤 Predict New Customers** | Upload any customer CSV → download with Prediction + Loan_Probability + Propensity_Tier columns |

## 🛠️ Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/universalbank-loan-dashboard.git
cd universalbank-loan-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set **Main file path** = `app.py`
4. Click **Deploy** — done in ~2 minutes!

## 📁 Files

```
universalbank_app/
├── app.py                      ← Main Streamlit application
├── UniversalBank.csv           ← Training dataset (5,000 customers)
├── sample_test_customers.csv   ← Sample file for the Predict tab
├── requirements.txt            ← Python dependencies
└── README.md                   ← This file
```

## 📊 Dataset

| Column | Description |
|---|---|
| Age | Customer age (years) |
| Experience | Professional experience (years, negative values clipped to 0) |
| Income | Annual income ($000) |
| ZIP Code | Home ZIP (dropped from model) |
| Family | Family size (1–4) |
| CCAvg | Monthly credit card spend ($000) |
| Education | 1=Undergrad, 2=Graduate, 3=Advanced/Professional |
| Mortgage | Mortgage value ($000) |
| **Personal Loan** | **Target: 1=Accepted, 0=Declined** |
| Securities Account | Has securities account (0/1) |
| CD Account | Has CD account (0/1) |
| Online | Uses internet banking (0/1) |
| CreditCard | Has bank credit card (0/1) |

## 🤖 Models

All models use `class_weight='balanced'` to handle the 90:10 class imbalance.

- **Decision Tree** — Interpretable baseline (max_depth=8)
- **Random Forest** — Ensemble of 300 trees (max_depth=12)
- **Gradient Boosted Tree** — Recommended for production scoring (200 estimators, lr=0.08)

## 💡 Key Insights

1. **Income** is the #1 predictor of loan acceptance
2. **CD Account holders** convert at 3× the rate of non-holders
3. **Family size 3–4 + Income >$120K** = 40–55% acceptance rate
4. Focus 75% of budget on **Platinum + Gold segments** for maximum ROI

---
Built for Universal Bank Marketing Team · Powered by Streamlit + scikit-learn + Plotly
