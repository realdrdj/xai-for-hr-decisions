import streamlit as st
import pandas as pd
import numpy as np
import shap
import lime.lime_tabular
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pathlib

st.set_page_config(page_title="XAI for HR Decisions", layout="wide")
st.title("XAI for HR Decisions (SHAP & LIME)")

# -------------------------
# 1. Upload or synthetic data
# -------------------------
uploaded_file = st.file_uploader("Upload HR dataset (CSV)", type="csv")

def generate_synthetic_hr(n=500, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(21, 60, n),
        "MonthlyIncome": rng.normal(12000, 4000, n).clip(3000, 30000).round(),
        "YearsAtCompany": rng.integers(0, 20, n),
        "JobLevel": rng.integers(1, 5, n),
        "OverTime": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "JobSatisfaction": rng.integers(1, 5, n),
        "BusinessTravel": rng.choice(["Non-Travel","Travel_Rarely","Travel_Frequently"], n, p=[0.2,0.6,0.2]),
        "Department": rng.choice(["Sales","R&D","HR"], n, p=[0.4,0.5,0.1]),
        "Gender": rng.choice(["Male","Female"], n, p=[0.55,0.45])
    })
    # Attrition label with simple rule
    risk = (df["OverTime"].eq("Yes")).astype(int)*0.8 + (df["JobSatisfaction"]<2).astype(int)*0.5 + rng.normal(0,0.3,n)
    df["Attrition"] = np.where(risk > 0.5, "Yes", "No")
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded.")
else:
    df = generate_synthetic_hr()
    st.info("No CSV uploaded. Using synthetic HR dataset.")

st.write("### Data preview", df.head())

# -------------------------
# 2. Preprocess & split
# -------------------------
target = "Attrition"
X = df.drop(columns=[target])
y = (df[target].astype(str).str.lower()=="yes").astype(int)

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# -------------------------
# 3. Model training
# -------------------------
model_choice = st.radio("Choose model:", ["Logistic Regression", "Random Forest"])
if model_choice == "Logistic Regression":
    model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=200))])
else:
    model = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])

model.fit(X_train, y_train)
st.success(f"{model_choice} trained.")

# -------------------------
# 4. SHAP Global Explanations
# -------------------------
st.subheader("Global Explanations (SHAP)")

# Use raw DataFrames (pipeline handles preprocessing internally)
explainer = shap.Explainer(model.predict_proba, X_train)

# Explain on sample (speed)
X_sample = X_test.sample(50, random_state=42)
shap_values = explainer(X_sample)

fig, ax = plt.subplots()
shap.summary_plot(
    shap_values[:,:,1],  # class 1 (Attrition=Yes)
    X_sample,
    feature_names=X_sample.columns,
    show=False
)
st.pyplot(fig)

# -------------------------
# 5. Local LIME Explanation
# -------------------------
st.subheader("Local Explanation (LIME)")
sample_id = st.slider("Select employee index", 0, len(X_test)-1, 0)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=preprocess.fit_transform(X_train),
    feature_names=preprocess.get_feature_names_out(),
    class_names=["No","Yes"],
    mode="classification"
)

exp = lime_explainer.explain_instance(
    preprocess.transform(X_test.iloc[[sample_id]])[0],
    model.predict_proba,
    num_features=5
)
st.write("LIME explanation for selected case:")
st.write(exp.as_list(label=1))

# -------------------------
# 6. Policy lever mapping
# -------------------------
st.subheader("Policy Levers (Example Mapping)")
LEVER_MAP = {
    "OverTime": "Review overtime policy; offer compensatory offs.",
    "BusinessTravel": "Rotate travel assignments; enable remote work.",
    "MonthlyIncome": "Audit salary bands; align with market.",
    "JobSatisfaction": "Manager coaching; recognition programs.",
    "YearsAtCompany": "Create mid-tenure growth pathways."
}

global_mean_abs = np.abs(shap_values[:,:,1].values).mean(axis=0)
feat_imp = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": global_mean_abs})
top_feats = feat_imp.sort_values("mean_abs_shap", ascending=False).head(5)

policy = [(f, LEVER_MAP.get(f.split("_")[0], "Define HR intervention")) for f in top_feats["feature"]]
st.table(pd.DataFrame(policy, columns=["Feature", "Suggested Action"]))

# -------------------------
# 7. Export HTML report
# -------------------------
if st.button("Generate Managerial Report"):
    html = f"""
    <html><body>
    <h2>Explainable AI HR Report</h2>
    <p><b>Model:</b> {model_choice}</p>
    <h3>Top Global Drivers</h3>
    {top_feats.to_html(index=False)}
    <h3>Policy Levers</h3>
    <ul>{''.join([f"<li>{f}: {a}</li>" for f,a in policy])}</ul>
    <p><i>Generated by Streamlit app</i></p>
    </body></html>
    """
    report_path = pathlib.Path("xai_hr_report.html")
    report_path.write_text(html, encoding="utf-8")
    st.download_button("Download Report", data=html, file_name="xai_hr_report.html", mime="text/html")

# -------------------------
# 8. Footer
# -------------------------
st.markdown(
    """
    <hr style="margin-top:50px; margin-bottom:10px;">
    <div style="text-align:center; font-size:14px;">
        Developed by <b>Prof. Dinesh K.</b> 
        <a href="https://linktr.ee/realdrdj" target="_blank">(link)</a>
    </div>
    """,
    unsafe_allow_html=True
)
