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
from typing import Tuple, List
import pathlib

# -------------------------
# Page & Title
# -------------------------
st.set_page_config(page_title="XAI for HR Decisions", layout="wide")
st.title("XAI for HR Decisions")

st.caption(
    "Interactive explanations for HR models using SHAP (global drivers) and LIME (case-specific reasons). "
    "Upload your CSV or use the demo dataset."
)

# -------------------------
# Helpers
# -------------------------
def make_ohe() -> OneHotEncoder:
    """Create an OHE that returns dense arrays, compatible across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # older sklearn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def to_dense(X):
    """Ensure we have a dense numpy array."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)

def friendly_name(name: str) -> str:
    """
    Turn encoded feature names like 'cat__OverTime_Yes' or 'num__MonthlyIncome'
    into human-readable labels: 'OverTime: Yes' or 'MonthlyIncome'.
    """
    base = name.split("__")[-1]  # drop 'cat__'/'num__'
    if "_" in base and not base.replace("_", "").isalpha():
        # when underscores are part of numeric bin names, skip prettifying
        return base
    parts = base.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}: {'_'.join(parts[1:])}"
    return base

def compute_shap(clf, X_train_enc: np.ndarray, X_test_enc: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Compute SHAP values for the fitted classifier using encoded/dense features.
    Returns (sv, mode) where sv has shape (n_samples, n_features) for class 1 if possible.
    mode is a short string with the explainer used.
    """
    # Prefer model-specific explainers
    if isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_enc)
        # TreeExplainer returns a list [class0, class1] for binary classification
        if isinstance(shap_values, list) and len(shap_values) >= 2:
            sv = shap_values[1]
        else:
            # Some versions return just one array; assume it's for the positive class
            sv = np.asarray(shap_values)
        return np.asarray(sv), "tree"
    elif isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_train_enc)
        sv = explainer.shap_values(X_test_enc)
        sv = np.asarray(sv)
        # Ensure 2D: (n_samples, n_features)
        if sv.ndim == 1:
            sv = sv.reshape(-1, X_test_enc.shape[1])
        return sv, "linear"
    else:
        # Fallback: model-agnostic (slower). Use predict_proba on encoded inputs.
        f = lambda data: clf.predict_proba(data)[:, 1]
        explainer = shap.Explainer(f, X_train_enc)
        sv = explainer(X_test_enc).values
        sv = np.asarray(sv)
        if sv.ndim == 1:
            sv = sv.reshape(-1, X_test_enc.shape[1])
        return sv, "agnostic"

def align_for_plot(sv: np.ndarray, X_enc: np.ndarray, feat_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Make sure sv, X_enc, and feat_names have matching n_features.
    Truncate or skip gracefully if there is still a mismatch.
    """
    n_feat = X_enc.shape[1]
    if sv.shape[1] != n_feat:
        # Align by truncation to the minimum common width
        min_feat = min(n_feat, sv.shape[1], len(feat_names))
        X_enc = X_enc[:, :min_feat]
        sv = sv[:, :min_feat]
        feat_names = list(feat_names[:min_feat])
    return sv, X_enc, feat_names

# -------------------------
# 1) Data input
# -------------------------
uploaded_file = st.file_uploader("Upload HR dataset (CSV)", type="csv")

def generate_synthetic_hr(n=500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Age": rng.integers(21, 60, n),
        "MonthlyIncome": rng.normal(12000, 4000, n).clip(3000, 30000).round(),
        "YearsAtCompany": rng.integers(0, 20, n),
        "JobLevel": rng.integers(1, 5, n),
        "OverTime": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "JobSatisfaction": rng.integers(1, 5, n),
        "BusinessTravel": rng.choice(["Non-Travel", "Travel_Rarely", "Travel_Frequently"], n, p=[0.2, 0.6, 0.2]),
        "Department": rng.choice(["Sales", "R&D", "HR"], n, p=[0.4, 0.5, 0.1]),
        "Gender": rng.choice(["Male", "Female"], n, p=[0.55, 0.45])
    })
    # Simple label rule to ensure signal
    risk = (
        (df["OverTime"].eq("Yes")).astype(int) * 0.8
        + (df["JobSatisfaction"] < 2).astype(int) * 0.5
        + rng.normal(0, 0.3, n)
    )
    df["Attrition"] = np.where(risk > 0.5, "Yes", "No")
    return df

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded.")
else:
    df = generate_synthetic_hr()
    st.info("No CSV uploaded. Using synthetic demo dataset.")

# Basic target normalization if user gives 0/1 rather than Yes/No
if "Attrition" in df.columns:
    if df["Attrition"].dropna().astype(str).str.lower().isin(["0", "1"]).all():
        df["Attrition"] = df["Attrition"].astype(int).map({0: "No", 1: "Yes"})

st.write("### 👀 Data preview")
st.dataframe(df.head(10), use_container_width=True)

# -------------------------
# 2) Preprocess & split
# -------------------------
target = "Attrition"
if target not in df.columns:
    st.error("Target column 'Attrition' not found in your data. Please include it as Yes/No or 1/0.")
    st.stop()

X = df.drop(columns=[target])
y = (df[target].astype(str).str.strip().str.lower() == "yes").astype(int)

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", make_ohe(), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y if y.nunique() == 2 else None, test_size=0.25, random_state=42
)

# -------------------------
# 3) Model training
# -------------------------
st.subheader("⚙️ Train model")
model_choice = st.radio("Choose learning algorithm:", ["Logistic Regression", "Random Forest"], horizontal=True)

if model_choice == "Logistic Regression":
    model = Pipeline([("prep", preprocess),
                      ("clf", LogisticRegression(max_iter=400, solver="lbfgs"))])
else:
    model = Pipeline([("prep", preprocess),
                      ("clf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))])

model.fit(X_train, y_train)
st.success(f"Model trained: {model_choice}")

# Use the fitted encoder from the trained pipeline (no re-fit drift)
prep = model.named_steps["prep"]
clf = model.named_steps["clf"]

X_train_enc = to_dense(prep.transform(X_train))
X_test_enc = to_dense(prep.transform(X_test))
feature_names = prep.get_feature_names_out()

friendly_feature_names = [friendly_name(f) for f in feature_names]

# -------------------------
# 4) SHAP: Global explanations
# -------------------------
st.subheader("🌍 Global drivers (SHAP)")
st.caption("Which features push attrition risk up or down across the dataset.")

top_feats = pd.DataFrame()
policy = []

try:
    sv, mode = compute_shap(clf, X_train_enc, X_test_enc)
    # Align shapes for plotting and tables
    sv, X_plot, fnames_plot = align_for_plot(sv, X_test_enc, np.array(friendly_feature_names))

    # Plot summary (bar = average impact)
    with st.spinner("Computing SHAP summary..."):
        fig, ax = plt.subplots()
        shap.summary_plot(sv, X_plot, feature_names=fnames_plot, show=False, plot_type="bar")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # Feature importance table
    global_mean_abs = np.abs(sv).mean(axis=0)
    feat_imp = pd.DataFrame({"feature": fnames_plot, "mean_abs_shap": global_mean_abs})
    top_feats = feat_imp.sort_values("mean_abs_shap", ascending=False).head(10)
    st.write("**Top global drivers**")
    st.dataframe(top_feats, use_container_width=True)

except Exception as e:
    st.error(f"SHAP could not be computed: {e}")

# -------------------------
# 5) LIME: Local explanation
# -------------------------
st.subheader("👤 Local reasons (LIME)")
st.caption("Pick an employee to see the top factors behind that specific prediction.")

if len(X_test_enc) > 0:
    sample_id = st.slider("Select row in test set", 0, len(X_test_enc) - 1, 0)
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_enc,
        feature_names=friendly_feature_names,
        class_names=["No", "Yes"],
        mode="classification",
        discretize_continuous=True
    )
    exp = lime_explainer.explain_instance(
        X_test_enc[sample_id],
        clf.predict_proba,
        num_features=8
    )
    st.write("**Top factors for this employee**")
    st.table(pd.DataFrame(exp.as_list(label=1), columns=["Factor", "Effect"]).head(8))
else:
    st.info("Not enough test rows to show a local explanation.")

# -------------------------
# 6) Policy levers
# -------------------------
st.subheader("🔧 Policy levers")
st.caption("Translate insights into actions. Draft levers shown for the most important features.")

LEVER_MAP = {
    "OverTime": "Review overtime scheduling and caps; offer compensatory offs.",
    "BusinessTravel": "Rotate travel assignments or increase remote collaboration.",
    "MonthlyIncome": "Audit salary bands; align to market percentiles.",
    "JobSatisfaction": "Manager coaching; recognition; role crafting.",
    "YearsAtCompany": "Mid-tenure growth pathways and skill rotations.",
}

if not top_feats.empty:
    # Map friendly names back to base feature before colon (e.g., 'OverTime: Yes' -> 'OverTime')
    base_feats = [f.split(":")[0] for f in top_feats["feature"]]
    policy = [(f, LEVER_MAP.get(f, "Define a targeted HR intervention.")) for f in base_feats[:5]]
    st.table(pd.DataFrame(policy, columns=["Driver (base feature)", "Suggested action"]))
else:
    st.info("Run SHAP successfully to populate policy levers.")

# -------------------------
# 7) Export HTML Managerial Report
# -------------------------
st.subheader("📥 Export managerial brief")
if st.button("Generate & download HTML report"):
    html = f"""
    <html>
    <head><meta charset="utf-8"><title>XAI HR Report</title></head>
    <body style="font-family:Arial, sans-serif; margin:24px;">
    <h2>XAI for HR Decisions — Managerial Brief</h2>
    <p><b>Model:</b> {model_choice}</p>
    <h3>Top global drivers (mean |SHAP|)</h3>
    {top_feats.to_html(index=False) if no
