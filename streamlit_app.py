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
        "BusinessTravel": rng.choice(
            ["Non-Travel","Travel_Rarely","Travel_Frequently"], 
            n, p=[0.2,0.6,0.2]),
        "Department": rng.choice(["Sales","R&D","HR"], n, p=[0.4,0.5,0.1]),
        "Gender": rng.choice(["Male","Female"], n, p=[0.55,0.45])
    })
    risk = (
        (df["OverTime"].eq("Yes")).astype(int)*0.8 
        + (df["JobSatisfaction"]<2).astype(int)*0.5 
        + rng.normal(0,0.3,n)
    )
    df["Attrition"] = np.where(risk > 0.5, "Yes", "No")
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded.")
else:
    df = generate_synthetic_hr()
    st.info("No CSV uploaded. Using synthetic HR dataset.")

st.write("### 👀 Data preview", df.head())

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

# -------------------------
# 3. Model training
# -------------------------
st.subheader("⚙️ Train Model")
model_choice = st.radio("Choose model:", ["Logistic Regression", "Random Forest"])
if model_choice == "Logistic Regression":
    model = Pipeline([("prep", preprocess), 
                      ("clf", LogisticRegression(max_iter=200))])
else:
    model = Pipeline([("prep", preprocess), 
                      ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])

model.fit(X_train, y_train)
st.success(f"{model_choice} trained successfully.")

# -------------------------
# 4. SHAP Global Explanations
# -------------------------
st.subheader("🌍 Global Explanations (SHAP)")
st.markdown("These plots show which features most strongly drive attrition risk in the **whole dataset**.")

X_train_enc = preprocess.fit_transform(X_train)
X_test_enc = preprocess.transform(X_test)
feature_names = preprocess.get_feature_names_out()
clf = model.named_steps["clf"]

top_feats = pd.DataFrame()
policy = []

try:
    if isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_enc)
        sv = shap_values[1]  # class 1 (Attrition=Yes)
    elif isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_train_enc)
        sv = explainer.shap_values(X_test_enc)
        if sv.ndim == 1:
            sv = sv.reshape(-1, len(feature_names))  # force 2D
    else:
        st.warning("Unsupported model type for SHAP.")
        sv = None

    if sv is not None:
        # Ensure alignment
        n_features = X_test_enc.shape[1]
        if sv.shape[1] != n_features:
            sv = sv[:, :n_features]

        # Plot
        fig, ax = plt.subplots()
        shap.summary_plot(sv, X_test_enc, feature_names=feature_names, show=False)
        st.pyplot(fig)

        # Global importance
        global_mean_abs = np.abs(sv).mean(axis=0)
        if len(global_mean_abs) == len(feature_names):
            feat_imp = pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": global_mean_abs
            }).sort_values("mean_abs_shap", ascending=False)
            top_feats = feat_imp.head(5)

except Exception as e:
    st.error(f"SHAP could not be computed: {e}")

# -------------------------
# 5. Local LIME Explanation
# -------------------------
st.subheader("👤 Local Explanation (LIME)")
st.markdown("Pick an employee to see which factors influenced *that specific prediction*.")

sample_id = st.slider("Select employee index", 0, len(X_test)-1, 0)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train_enc,
    feature_names=feature_names,
    class_names=["No","Yes"],
    mode="classification"
)

exp = lime_explainer.explain_instance(
    X_test_enc[sample_id],
    clf.predict_proba,
    num_features=5
)
st.write("Top factors for this case:")
st.write(exp.as_list(label=1))

# -------------------------
# 6. Policy lever mapping
# -------------------------
st.subheader("🔧 Policy Levers (Turn insights into actions)")
LEVER_MAP = {
    "OverTime": "Review overtime policy; offer compensatory offs.",
    "BusinessTravel": "Rotate travel assignments; enable remote work.",
    "MonthlyIncome": "Audit salary bands; align with market.",
    "JobSatisfaction": "Manager coaching; recognition programs.",
    "YearsAtCompany": "Create mid-tenure growth pathways."
}

if not top_feats.empty:
    policy = [(f, LEVER_MAP.get(f.split("_")[0], "Define HR intervention")) for f in top_feats["feature"]]
    st.table(pd.DataFrame(policy, columns=["Feature", "Suggested Action"]))

# -------------------------
# 7. Export HTML report
# -------------------------
if st.button("📥 Generate Managerial Report"):
    html = f"""
    <html><body>
    <h2>Explainable AI HR Report</h2>
    <p><b>Model:</b> {model_choice}</p>
    <h3>Top Global Drivers</h3>
    {top_feats.to_html(index=False) if not top_feats.empty else 'Not available'}
    <h3>Policy Levers</h3>
    <ul>{''.join([f"<li>{f}: {a}</li>" for f,a in policy]) if policy else 'Not available'}</ul>
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
