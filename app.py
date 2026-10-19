import warnings; warnings.filterwarnings("ignore")
import os, zipfile
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
import plotly.express as px
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve, classification_report)

sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)
os.makedirs("figs", exist_ok=True)
def H(t): print("\n" + "="*75 + "\n" + t + "\n" + "="*75)
def save(name):
    plt.savefig(f"figs/{name}.png", dpi=200, bbox_inches="tight"); plt.show()

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
raw = pd.read_csv(URL)

# ============================ WEEK 1 ============================
H("WEEK 1: RAW DATA INSPECTION")
print("Shape:", raw.shape)
print(raw.dtypes.to_string())
print("\nNulls (isnull) in raw:\n", raw.isnull().sum()[raw.isnull().sum() > 0])
blank = (raw["TotalCharges"].astype(str).str.strip() == "")
print("Blank-string TotalCharges count:", blank.sum())
print("Full-row duplicates:", raw.duplicated().sum(), "| Duplicate customerIDs:", raw["customerID"].duplicated().sum())

df = raw.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nNaN after to_numeric:", df["TotalCharges"].isnull().sum())
print("Rows with NaN TotalCharges (tenure, MonthlyCharges, Churn):")
print(df[df["TotalCharges"].isnull()][["customerID","tenure","MonthlyCharges","Churn"]].to_string())
print("Tenure values among those rows:", df[df["TotalCharges"].isnull()]["tenure"].unique())

df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0:"No", 1:"Yes"})
obj_cols = df.select_dtypes("object").columns
df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
df["Churn_flag"] = (df["Churn"] == "Yes").astype(int)
print("\nNulls after cleaning:", df.isnull().sum().sum())
print("Unique values of categorical columns:")
for c in obj_cols.drop("customerID"):
    print(f"  {c}: {list(df[c].unique())}")

H("WEEK 1: OUTLIERS (IQR rule)")
for col in ["tenure","MonthlyCharges","TotalCharges"]:
    q1, q3 = df[col].quantile([.25,.75]); iqr = q3 - q1
    n = ((df[col] < q1-1.5*iqr) | (df[col] > q3+1.5*iqr)).sum()
    print(f"{col}: IQR outliers = {n}, skew = {df[col].skew():.3f}")

H("WEEK 1: SUMMARY STATISTICS")
print(df[["tenure","MonthlyCharges","TotalCharges"]].describe().round(2).to_string())
print("\nMedian:", df[["tenure","MonthlyCharges","TotalCharges"]].median().round(2).to_dict())
print("\nChurn counts:\n", df["Churn"].value_counts().to_string())
print("Churn %:\n", (df["Churn"].value_counts(normalize=True)*100).round(2).to_string())
print("\nCorrelation matrix:\n", df[["tenure","MonthlyCharges","TotalCharges","Churn_flag"]].corr().round(3).to_string())

H("WEEK 1: CHURN RATE (%) BY CATEGORY")
for c in ["Contract","InternetService","PaymentMethod","TechSupport","OnlineSecurity","SeniorCitizen",
          "Partner","Dependents","gender","PaperlessBilling","StreamingTV","MultipleLines"]:
    print(f"\n{c}:\n", (df.groupby(c)["Churn_flag"].mean()*100).round(2).to_string())

df.to_csv("telco_clean.csv", index=False)

# W1 figures
plt.figure(figsize=(10,4)); r2 = raw.copy(); r2["TotalCharges"] = pd.to_numeric(r2["TotalCharges"], errors="coerce")
sns.heatmap(r2.isnull(), cbar=False, yticklabels=False); plt.title("Missing Values Map (before cleaning)"); save("w1_fig1_missing")
fig, ax = plt.subplots(1,3, figsize=(15,4))
for a, col in zip(ax, ["tenure","MonthlyCharges","TotalCharges"]):
    sns.histplot(df[col], kde=True, ax=a); a.set_title(f"Distribution of {col}")
save("w1_fig2_hist")
plt.figure(figsize=(6,5)); sns.boxplot(data=df, x="Churn", y="MonthlyCharges"); plt.title("Monthly Charges by Churn"); save("w1_fig3_box")
(df.groupby("Contract")["Churn_flag"].mean()*100).plot(kind="bar", figsize=(6,4), color="#d1495b")
plt.ylabel("Churn rate (%)"); plt.title("Churn Rate by Contract Type"); plt.xticks(rotation=0); save("w1_fig4_contract")
plt.figure(figsize=(6,5)); sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges","Churn_flag"]].corr(), annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Matrix"); save("w1_fig5_corr")

# ============================ WEEK 2 ============================
H("WEEK 2: STORY NUMBERS")
COL = {"Yes":"#d1495b", "No":"#4c78a8"}
df["tenure_band"] = pd.cut(df["tenure"], [-1,6,12,24,48,72], labels=["0-6","7-12","13-24","25-48","49-72"])
t = df.groupby("tenure_band", observed=True)["Churn_flag"].mean().mul(100)
print("Churn % by tenure band:\n", t.round(2).to_string())
print("\nChurn % by contract:\n", pd.crosstab(df["Contract"], df["Churn"], normalize="index").mul(100).round(2).to_string())
pv = df.pivot_table(index="InternetService", columns="Contract", values="Churn_flag", aggfunc="mean").mul(100)
print("\nChurn % InternetService x Contract:\n", pv.round(1).to_string())
print("\nMean MonthlyCharges by churn:", df.groupby("Churn")["MonthlyCharges"].mean().round(2).to_dict())
sub = df[df["InternetService"] != "No"]
rows = []
for s in ["OnlineSecurity","TechSupport","OnlineBackup","DeviceProtection"]:
    r = sub.groupby(s)["Churn_flag"].mean().mul(100)
    rows.append({"Service":s, "Has service":r.get("Yes"), "No service":r.get("No")})
svc_df = pd.DataFrame(rows).set_index("Service"); print("\nAdd-on churn % (internet customers):\n", svc_df.round(2).to_string())

counts = df["Churn"].value_counts()
fig, ax = plt.subplots(figsize=(5,5)); ax.pie(counts, labels=counts.index, colors=[COL[i] for i in counts.index], autopct="%1.1f%%", wedgeprops=dict(width=0.4))
ax.set_title("About 1 in 4 customers has left"); save("w2_v1_donut")
plt.figure(figsize=(7,4)); plt.plot(t.index.astype(str), t.values, marker="o", color=COL["Yes"], lw=2.5)
for x, y in zip(t.index.astype(str), t.values): plt.text(x, y+1, f"{y:.0f}%", ha="center")
plt.title("Churn is highest in the first months and falls with loyalty"); plt.ylabel("Churn rate (%)"); plt.xlabel("Months as customer"); save("w2_v2_tenure")
ct_ = pd.crosstab(df["Contract"], df["Churn"], normalize="index").mul(100)
ct_[["No","Yes"]].plot(kind="barh", stacked=True, color=[COL["No"],COL["Yes"]], figsize=(8,3.5))
plt.title("Month-to-month customers leave far more often than contracted ones"); plt.xlabel("% of customers"); save("w2_v3_contract")
plt.figure(figsize=(7,4))
for k, g in df.groupby("Churn"): sns.kdeplot(g["MonthlyCharges"], fill=True, alpha=.4, color=COL[k], label=f"Churn = {k}")
plt.title("Churners are concentrated at higher monthly bills"); plt.legend(); save("w2_v4_price")
plt.figure(figsize=(7,4)); sns.heatmap(pv, annot=True, fmt=".0f", cmap="Reds", cbar_kws={"label":"Churn %"})
plt.title("Highest risk: fiber optic on month-to-month"); save("w2_v5_heatmap")
svc_df.plot(kind="bar", figsize=(8,4), color=["#59a14f","#d1495b"]); plt.title("Customers with security and support services churn less")
plt.ylabel("Churn rate (%)"); plt.xticks(rotation=0); save("w2_v6_services")
plt.figure(figsize=(7,5)); sns.scatterplot(data=df, x="tenure", y="MonthlyCharges", hue="Churn", palette=COL, alpha=.5)
plt.title("Every customer: tenure vs monthly bill"); save("w2_v7_scatter")
px.scatter(df, x="tenure", y="MonthlyCharges", color="Churn", color_discrete_map=COL, opacity=.6,
           hover_data=["Contract","InternetService"]).write_html("figs/w2_v7_interactive.html")

# ============================ WEEK 3 ============================
H("WEEK 3: HYPOTHESIS TESTS (alpha = 0.05, Bonferroni = 0.0125)")
a = df.loc[df.Churn=="Yes","MonthlyCharges"]; b = df.loc[df.Churn=="No","MonthlyCharges"]
print(f"H1: mean churned = {a.mean():.2f} (sd {a.std():.2f}, n={len(a)}), retained = {b.mean():.2f} (sd {b.std():.2f}, n={len(b)})")
print("Shapiro-type normality (D'Agostino) churned:", stats.normaltest(a), "retained:", stats.normaltest(b))
print("Levene:", stats.levene(a, b))
tt, p = stats.ttest_ind(a, b, equal_var=False); print(f"Welch t = {tt:.4f}, p = {p:.3e}")
diff = a.mean()-b.mean(); va, vb = a.var(ddof=1)/len(a), b.var(ddof=1)/len(b); se = np.sqrt(va+vb)
dof = (va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1)); ci = (diff-stats.t.ppf(.975,dof)*se, diff+stats.t.ppf(.975,dof)*se)
sp = np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
print(f"Diff = {diff:.3f}, 95% CI = ({ci[0]:.3f}, {ci[1]:.3f}), Cohen's d = {diff/sp:.3f}, Welch dof = {dof:.1f}")
print("Mann-Whitney:", stats.mannwhitneyu(a, b, alternative="two-sided"))

ctab = pd.crosstab(df["Contract"], df["Churn"]); chi2, p, dofc, exp_c = stats.chi2_contingency(ctab)
print(f"\nH2: Contract x Churn\n{ctab.to_string()}\nchi2 = {chi2:.3f}, dof = {dofc}, p = {p:.3e}, min expected = {exp_c.min():.1f}")
print(f"Cramer's V = {np.sqrt(chi2/(ctab.values.sum()*(min(ctab.shape)-1))):.4f}")
print("Row %:\n", ctab.div(ctab.sum(axis=1), axis=0).mul(100).round(2).to_string())

groups = [g["tenure"].values for _, g in df.groupby("Contract")]
print("\nH3: tenure by contract. Means:", df.groupby("Contract")["tenure"].mean().round(2).to_dict())
print("Levene:", stats.levene(*groups)); F, p = stats.f_oneway(*groups); print(f"ANOVA F = {F:.3f}, p = {p:.3e}")
aov = sm.stats.anova_lm(ols("tenure ~ C(Contract)", data=df).fit(), typ=2); print(aov.round(4).to_string())
print(f"eta squared = {aov['sum_sq'].iloc[0]/aov['sum_sq'].sum():.4f}")
print(pairwise_tukeyhsd(df["tenure"], df["Contract"]).summary())
print("Kruskal-Wallis:", stats.kruskal(*groups))

ct2 = pd.crosstab(df["PaymentMethod"], df["Churn"]); chi2b, pb, dofb, _ = stats.chi2_contingency(ct2)
print(f"\nH4: Payment x Churn\nchi2 = {chi2b:.3f}, dof = {dofb}, p = {pb:.3e}, Cramer's V = {np.sqrt(chi2b/(ct2.values.sum()*(min(ct2.shape)-1))):.4f}")
print("Row %:\n", ct2.div(ct2.sum(axis=1), axis=0).mul(100).round(2).to_string())

# W3 figures
plt.figure(figsize=(7,4)); sns.histplot(data=df, x="MonthlyCharges", hue="Churn", stat="density", common_norm=False, alpha=.5)
plt.axvline(a.mean(), color="C1", ls="--"); plt.axvline(b.mean(), color="C0", ls="--"); plt.title("Monthly charges by churn (dashed = means)"); save("w3_fig1_dist")
plt.figure(figsize=(5,4)); plt.errorbar([0],[diff], yerr=[[diff-ci[0]],[ci[1]-diff]], fmt="o", capsize=8); plt.axhline(0, color="gray", ls="--")
plt.xticks([]); plt.ylabel("Mean diff in monthly charge (churned - retained)"); plt.title("95% CI of the difference"); save("w3_fig2_ci")
resid = (ctab.values - exp_c)/np.sqrt(exp_c)
sns.heatmap(pd.DataFrame(resid, index=ctab.index, columns=ctab.columns), annot=True, fmt=".1f", cmap="coolwarm", center=0)
plt.title("Standardized residuals: contract vs churn"); save("w3_fig3_resid")
sns.boxplot(data=df, x="Contract", y="tenure"); plt.title("Tenure by contract type"); save("w3_fig4_box")

# ============================ WEEK 4 ============================
H("WEEK 4: MODELLING")
d4 = df.drop(columns=["customerID","tenure_band"]).copy()
for c in ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
    d4[c] = d4[c].replace({"No internet service":"No","No phone service":"No"})
y = d4["Churn_flag"]; X = d4.drop(columns=["Churn","Churn_flag"])
num = ["tenure","MonthlyCharges","TotalCharges"]; cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train/test sizes:", len(X_tr), len(X_te), "| churn rate train/test:", round(y_tr.mean(),4), round(y_te.mean(),4))
print("Baseline accuracy (predict all 'No'):", round(1-y_te.mean(),4))
models = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
          "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
          "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1)}
pipes = {n: Pipeline([("pre", pre), ("clf", m)]) for n, m in models.items()}
skf = StratifiedKFold(5, shuffle=True, random_state=42); rows = []
for n, p_ in pipes.items():
    cv = cross_val_score(p_, X_tr, y_tr, cv=skf, scoring="roc_auc"); p_.fit(X_tr, y_tr)
    pred = p_.predict(X_te); proba = p_.predict_proba(X_te)[:,1]
    print(f"{n}: CV AUC = {cv.mean():.3f} +/- {cv.std():.3f}")
    rows.append({"Model":n, "Accuracy":accuracy_score(y_te,pred), "Precision":precision_score(y_te,pred), "Recall":recall_score(y_te,pred),
                 "F1":f1_score(y_te,pred), "ROC-AUC":roc_auc_score(y_te,proba), "Train acc":accuracy_score(y_tr,p_.predict(X_tr)),
                 "CV AUC":cv.mean()})
res = pd.DataFrame(rows).set_index("Model").round(3); print("\n", res.to_string())
for n, p_ in pipes.items():
    tn, fp, fn, tp = confusion_matrix(y_te, p_.predict(X_te)).ravel()
    print(f"{n} confusion: TN={tn} FP={fp} FN={fn} TP={tp} | FPR={fp/(fp+tn):.3f} FNR={fn/(fn+tp):.3f}")
print("\nLogistic Regression report:\n", classification_report(y_te, pipes["Logistic Regression"].predict(X_te), target_names=["Stay","Churn"]))

lr = pipes["Logistic Regression"]; proba = lr.predict_proba(X_te)[:,1]
print("Threshold tuning (LR):")
for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pr_ = (proba >= th).astype(int)
    print(f"  t={th}: precision={precision_score(y_te,pr_):.3f} recall={recall_score(y_te,pr_):.3f} f1={f1_score(y_te,pr_):.3f}")
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["pre"].get_feature_names_out()).sort_values()
print("\nTop negative coefficients:\n", coef.head(8).round(3).to_string()); print("Top positive coefficients:\n", coef.tail(8).round(3).to_string())
print("Odds ratios (top 5 positive):\n", np.exp(coef.tail(5)).round(2).to_string())
rf_imp = pd.Series(pipes["Random Forest"].named_steps["clf"].feature_importances_, index=pipes["Random Forest"].named_steps["pre"].get_feature_names_out()).sort_values(ascending=False)
print("\nRandom Forest top 8 importances:\n", rf_imp.head(8).round(3).to_string())
gs = GridSearchCV(lr, {"clf__C":[0.01,0.1,1,10]}, cv=skf, scoring="roc_auc").fit(X_tr, y_tr)
print("\nGridSearch best params:", gs.best_params_, "best CV AUC:", round(gs.best_score_,4), "| test AUC:", round(roc_auc_score(y_te, gs.predict_proba(X_te)[:,1]),4))

fig, ax = plt.subplots(1,3, figsize=(15,4))
for a_, (n, p_) in zip(ax, pipes.items()):
    ConfusionMatrixDisplay.from_estimator(p_, X_te, y_te, ax=a_, cmap="Blues", display_labels=["Stay","Churn"], colorbar=False); a_.set_title(n)
save("w4_fig1_confusion")
plt.figure(figsize=(6,5))
for n, p_ in pipes.items():
    pb_ = p_.predict_proba(X_te)[:,1]; fpr, tpr, _ = roc_curve(y_te, pb_); plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y_te,pb_):.2f})")
plt.plot([0,1],[0,1],"k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.legend(); plt.title("ROC curves"); save("w4_fig2_roc")
plt.figure(figsize=(6,5))
for n, p_ in pipes.items():
    pr_, rc_, _ = precision_recall_curve(y_te, p_.predict_proba(X_te)[:,1]); plt.plot(rc_, pr_, label=n)
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.title("Precision-Recall curves"); save("w4_fig3_pr")
pd.concat([coef.head(8), coef.tail(8)]).plot(kind="barh", figsize=(7,6)); plt.title("Strongest drivers (logistic regression)"); save("w4_fig4_coef")
sizes, trs, vas = learning_curve(pipes["Random Forest"], X_tr, y_tr, cv=5, scoring="roc_auc", train_sizes=np.linspace(.1,1,6), n_jobs=-1)
plt.figure(figsize=(6,4)); plt.plot(sizes, trs.mean(1), label="Train"); plt.plot(sizes, vas.mean(1), label="Validation")
plt.legend(); plt.xlabel("Training examples"); plt.ylabel("ROC-AUC"); plt.title("Learning curve (Random Forest)"); save("w4_fig5_learning")
print("Learning curve final train/val AUC:", round(trs.mean(1)[-1],3), round(vas.mean(1)[-1],3))

# ============================ WEEK 5 ============================
H("WEEK 5: BUSINESS IMPACT NUMBERS")
ch = df[df.Churn=="Yes"]
print("Customers:", len(df), "| churned:", len(ch), f"({len(ch)/len(df)*100:.2f}%)")
print("Avg monthly charge churned / retained:", round(ch["MonthlyCharges"].mean(),2), round(df[df.Churn=="No"]["MonthlyCharges"].mean(),2))
print("Monthly revenue lost:", round(ch["MonthlyCharges"].sum(),2), "| Annual recurring revenue lost:", round(ch["MonthlyCharges"].sum()*12,2))
for r in [0.10, 0.20, 0.30]:
    print(f"  {int(r*100)}% fewer churners -> {int(len(ch)*r)} customers, annual revenue retained = {ch['MonthlyCharges'].sum()*12*r:,.0f}")
print("Share of churners on month-to-month:", round((ch["Contract"]=="Month-to-month").mean()*100,1), "%")
print("Share of churners with tenure <= 12:", round((ch["tenure"]<=12).mean()*100,1), "%")
print("Share of churners on fiber:", round((ch["InternetService"]=="Fiber optic").mean()*100,1), "%")
print("Share of churners paying by electronic check:", round((ch["PaymentMethod"]=="Electronic check").mean()*100,1), "%")
seg = df[(df.Contract=="Month-to-month") & (df.InternetService=="Fiber optic") & (df.tenure<=12)]
print(f"High-risk segment (M2M + fiber + tenure<=12): n={len(seg)}, churn rate={seg['Churn_flag'].mean()*100:.1f}%")
seg2 = df[(df.Contract=="Month-to-month") & (df.PaymentMethod=="Electronic check")]
print(f"M2M + electronic check: n={len(seg2)}, churn rate={seg2['Churn_flag'].mean()*100:.1f}%")
print("Share of customers on month-to-month:", round((df.Contract=='Month-to-month').mean()*100,1), "%")

# package
df.to_csv("figs/telco_clean.csv", index=False)
with zipfile.ZipFile("churn_figures.zip", "w") as z:
    for f in os.listdir("figs"): z.write(f"figs/{f}")
try:
    from google.colab import files; files.download("churn_figures.zip")
except Exception: print("Zip saved as churn_figures.zip in the file panel.")
print("\nDONE. Copy ALL the text output above and paste it to Claude.")

#PART2:
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
raw = pd.read_csv(URL); df = raw.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn_flag"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0:"No", 1:"Yes"})

# 1) full normality p-values
a = df.loc[df.Churn=="Yes","MonthlyCharges"]; b = df.loc[df.Churn=="No","MonthlyCharges"]
print("Normality p churned:", stats.normaltest(a).pvalue, "| retained:", stats.normaltest(b).pvalue)

# 2) Week 5 business numbers
ch = df[df.Churn_flag==1]
print("Monthly revenue lost:", round(ch["MonthlyCharges"].sum(),2), "| Annual:", round(ch["MonthlyCharges"].sum()*12,2))
for r in [0.10,0.20,0.30]:
    print(f"{int(r*100)}% fewer churners -> {int(len(ch)*r)} customers, annual revenue retained = {ch['MonthlyCharges'].sum()*12*r:,.0f}")
print("Churners on month-to-month %:", round((ch.Contract=="Month-to-month").mean()*100,1))
print("Churners with tenure<=12 %:", round((ch.tenure<=12).mean()*100,1))
print("Churners on fiber %:", round((ch.InternetService=="Fiber optic").mean()*100,1))
print("Churners paying electronic check %:", round((ch.PaymentMethod=="Electronic check").mean()*100,1))
seg = df[(df.Contract=="Month-to-month")&(df.InternetService=="Fiber optic")&(df.tenure<=12)]
print(f"Segment M2M+fiber+tenure<=12: n={len(seg)}, churn={seg.Churn_flag.mean()*100:.1f}%")
seg2 = df[(df.Contract=="Month-to-month")&(df.PaymentMethod=="Electronic check")]
print(f"Segment M2M+echeck: n={len(seg2)}, churn={seg2.Churn_flag.mean()*100:.1f}%")
inet = df[df.InternetService!="No"]
print("Mean monthly charge among internet customers, churned vs retained:",
      inet.groupby("Churn")["MonthlyCharges"].mean().round(2).to_dict())

# 3) learning curve numbers (Random Forest)
d4 = df.drop(columns=["customerID","Churn"]).copy()
for c in ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]:
    d4[c] = d4[c].replace({"No internet service":"No","No phone service":"No"})
y = d4["Churn_flag"]; X = d4.drop(columns=["Churn_flag"])
num = ["tenure","MonthlyCharges","TotalCharges"]; cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([("num",StandardScaler(),num),("cat",OneHotEncoder(drop="if_binary",handle_unknown="ignore"),cat)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rf = Pipeline([("pre",pre),("clf",RandomForestClassifier(n_estimators=300,max_depth=8,class_weight="balanced",random_state=42,n_jobs=-1))])
sizes, trs, vas = learning_curve(rf, X_tr, y_tr, cv=5, scoring="roc_auc", train_sizes=np.linspace(.1,1,6), n_jobs=-1)
print("Learning curve sizes:", sizes); print("Train AUC:", trs.mean(1).round(3)); print("Val AUC:", vas.mean(1).round(3))

# 4) clearer Figure 1 for Week 1
r_ = raw.copy(); r_["TC"] = pd.to_numeric(r_["TotalCharges"], errors="coerce"); miss = r_[r_["TC"].isna()]
plt.figure(figsize=(7,4.5))
plt.scatter(r_["tenure"], r_["TC"], s=6, alpha=.3, label="Customers with a valid TotalCharges")
plt.scatter(miss["tenure"], [0]*len(miss), color="red", s=70, label="Blank TotalCharges (11 customers)")
plt.xlabel("Tenure (months)"); plt.ylabel("TotalCharges"); plt.legend()
plt.title("The 11 'missing' totals all belong to customers with tenure = 0")
plt.savefig("w1_fig1b_blank_totalcharges.png", dpi=200, bbox_inches="tight"); plt.show()
