# Telecom Customer Churn Prediction and Analysis
> End-to-end machine learning pipeline to predict customer churn using Decision Tree and Neural Network classifiers — with hyperparameter tuning, model explainability, and a Power BI dashboard.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Faara34/TelecomChurnMl/blob/main/CwMl.ipynb)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-green)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-yellow)

---

## Project Overview

Customer churn — when subscribers cancel their service — directly impacts revenue. This project builds and compares two classification models to identify customers at risk of churning, enabling businesses to take proactive retention action.

The full pipeline covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Model training with hyperparameter tuning
- Evaluation using multiple metrics
- Interactive Power BI dashboard for business stakeholders

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

| Property | Value |
|---|---|
| Rows | 7,043 |
| Columns | 21 |
| Target Variable | `Churn` (Yes / No) |
| Class Distribution | 73.46% Not Churned / 26.54% Churned |

**Key features:** tenure, MonthlyCharges, TotalCharges, Contract type, InternetService, PaymentMethod, and 15+ categorical service features.

---

## 🔍 Exploratory Data Analysis

- Visualized churn distribution using count plots
- Plotted histograms and boxplots for numerical features (tenure, MonthlyCharges, TotalCharges) by churn status
- Scatter plot of TotalCharges vs MonthlyCharges coloured by churn — revealed partial separability
- Count plots for all categorical features with churn hue — identified Contract type, InternetService, and PaymentMethod as strong churn signals
- Correlation heatmap — tenure showed a strong negative correlation with churn (-0.35); MonthlyCharges showed a positive correlation (0.19)

**Key EDA Findings:**
- Month-to-Month contract customers churn at a significantly higher rate
- Customers with Fiber Optic internet churn more than DSL or No service customers
- Electronic check is the payment method most associated with churn
- Short-tenure customers are disproportionately likely to churn

---

## ⚙️ Data Preprocessing

| Step | Detail |
|---|---|
| Missing values | TotalCharges had 11 nulls — imputed with column median |
| Duplicates | None found |
| Irrelevant features | CustomerID dropped to prevent data leakage |
| Binary encoding | Label Encoding for 2-category columns |
| Multi-category encoding | One-Hot Encoding with `drop_first=True` to avoid multicollinearity |
| Feature scaling | StandardScaler (mean=0, std=1) — required for Neural Network |
| Train/Test split | 80% train / 20% test, stratified on target |

---

## Models

### 1. Decision Tree Classifier
- Serves as an interpretable baseline
- Tuned using **GridSearchCV** (5-fold cross-validation)
- Parameters searched: `max_depth`, `min_samples_split`, `criterion` (gini/entropy)

### 2. Neural Network (TensorFlow / Keras)
- Captures non-linear patterns in the data
- Tuned using **KerasTuner (Hyperband)**
- Parameters searched: units, dropout rate, learning rate, optimizer
- Best model selected based on validation loss

---

## Results

| Metric | Decision Tree | Neural Network |
|---|---|---|
| Accuracy | 78.71% | 78.57% |
| Precision | 66.82% | 59.18% |
| Recall | 39.30% | 62.03% |
| F1-Score | 49.49% | 60.57% |
| ROC-AUC | **0.8168** | **0.8273** |

**Key takeaway:** The Neural Network outperforms the Decision Tree on Recall, F1-Score, and ROC-AUC — meaning it is better at identifying customers who will actually churn. The Decision Tree remains valuable as an interpretable baseline for business stakeholders.

---

##  Project Structure

```
TelecomChurnMl/
│
├── CwMl.ipynb              # Main notebook (EDA + Preprocessing + Models)
├── churnanalysis.pbix      # Power BI dashboard
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (download from Kaggle)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualisation |
| Scikit-learn | Decision Tree, preprocessing, metrics |
| TensorFlow / Keras | Neural Network |
| KerasTuner | Neural Network hyperparameter tuning |
| Power BI | Business dashboard |
| Google Colab | Development environment |

---

##  How to Run

1. Clone the repository
```bash
git clone https://github.com/Faara34/TelecomChurnMl.git
cd TelecomChurnMl
```

2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow keras-tuner
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place it in the project folder

4. Open the notebook
```bash
jupyter notebook CwMl.ipynb
```
Or open directly in Google Colab using the badge at the top.

---

## Future Enhancements

- [ ] Handle class imbalance using SMOTE or class-weighted loss functions
- [ ] Add SHAP explainability for Neural Network predictions
- [ ] Engineer new features (e.g. charge-per-tenure ratio, service count)
- [ ] Ensemble Decision Tree + Neural Network via stacking
- [ ] Deploy as a real-time prediction API using FastAPI or Streamlit

---

## Ethical Considerations

- CustomerID removed from training to prevent personal data leakage
- Multiple metrics used beyond accuracy to ensure fair evaluation on imbalanced classes
- Random seeds fixed for reproducibility and reviewability
- Model predictions will be reviewed across demographic groups to detect unfair bias

---


## 📄 License

This project was developed as academic coursework for CM2604 Machine Learning at RGU.
