# Telecom Customer Churn: End-to-End Data Science Project

A five-week data science project on customer churn using the public IBM Telco Customer Churn dataset (7,043 customers, 21 columns). It covers data cleaning, exploratory analysis, visual storytelling, statistical hypothesis testing, machine learning and strategic recommendations.

## Repository contents

- `churn_analysis.ipynb` : full Python code (cleaning, EDA, visualizations, hypothesis tests, models, business impact numbers)
- `telco_clean.csv` : cleaned dataset used in all later stages
- `Project_Results.pdf` : notebook output with all graphs and results
- `figures/` : saved charts (PNG) used in the reports
- `README.md` : this file

## Dataset

IBM Telco Customer Churn, loaded directly from a public URL:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

## What was done

1. Data cleaning and EDA: found that `TotalCharges` was stored as text with 11 blank entries (all customers with zero tenure, not yet billed), converted the type, filled with 0, checked duplicates and outliers (none found), and explored distributions and correlations.
2. Visual storytelling: seven visualizations (donut, line, stacked bar, density, heatmap, grouped bar, scatter) built for a non-technical audience.
3. Hypothesis testing: Welch t-test, chi-square tests with Cramer's V, one-way ANOVA with Tukey HSD, plus Mann-Whitney and Kruskal-Wallis robustness checks, with Bonferroni correction.
4. Machine learning: logistic regression, decision tree and random forest in a leakage-free scikit-learn pipeline, evaluated with accuracy, precision, recall, F1, ROC-AUC, cross-validation, threshold tuning and a learning curve.
5. Final report: synthesis of all stages, strategic recommendations and business impact estimates.

## Key results

- Overall churn rate: 26.54 percent (1,869 of 7,043 customers).
- Churn by contract: 42.71 percent month-to-month, 11.27 percent one-year, 2.83 percent two-year (Cramer's V = 0.41).
- Churn falls from 52.9 percent in the first 6 months to 9.5 percent after 49 months.
- Fiber optic customers churn at 41.89 percent versus 18.96 percent for DSL; electronic check payers churn at 45.29 percent.
- Higher average bills among churners are mostly a plan-mix effect: among internet customers, churned and retained customers pay almost the same (77.92 vs 76.34).
- Model performance (test set): logistic regression ROC-AUC 0.842, recall 0.786, precision 0.505. Decision tree and random forest reach about 0.84 as well.
- Estimated recurring revenue lost to churn: about 1.67 million per year; a 10 percent reduction in churners would retain about 167,000 per year (illustrative).

## How to run

1. Clone the repository.
2. Install dependencies:
   `pip install pandas numpy matplotlib seaborn plotly scipy statsmodels scikit-learn`
3. Open `churn_analysis.ipynb` in Jupyter or Google Colab and run all cells. The data downloads automatically from the URL above, and figures are saved to `figures/`.

## Tools

Python, pandas, NumPy, Matplotlib, Seaborn, Plotly, SciPy, Statsmodels, scikit-learn, Google Colab.

## Limitations

The data is observational and a single snapshot, so findings show associations, not causes. It has no usage, complaint or competitor data, which likely limits model performance at an AUC of about 0.84. Recommendations should be validated with controlled pilots.
