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

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="XAI for HR Decisions", layout="wide")
st.title("XAI for HR Decisions (SHAP & LIME)")
st.caption("Upload HR data or use the demo. SHAP shows global drivers; LIME explains individual employees.")

# -------------------------
# Helpers
# -------------------------
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def to_dense_float(X):
    if hasattr(X, "todense"):
        return np.asarray(X.todense()).astype("float32")
    return np.asarray(X).astype("float32")

def friendly_name(name: str) -> str:
    """Make encoded feature names more readable."""
    base = name.split("__")[-1]
    parts = base.split("_")
    if len(parts) > 1:
        return f"{parts[0]}: {' '.join(parts[1:])}"
    return base

# -------------------------
# Synthetic Dataset
# -------------------------
def generate_synthetic_hr(n=200, seed=42):
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
            n, p=[0.2,0.6,0.2]
        ),
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

# -------------------------
# Template Download
# -------------------------
template = pd.DataFrame({
    "Age":[30],
    "MonthlyIncome":[12000],
    "YearsAtCompany":[5],
    "JobLevel":[2],
    "OverTime":["Yes"],
    "JobSatisfaction":[3],
    "BusinessTravel":["Travel_Rarely"],
    "Department":["Sales"],
    "Gender":["Male"],
    "Attrition":["Yes"]
})
st.download_button(
    "📥 Download HR CSV Template",
    template.to_csv(index=False),
    file_name="hr_template.csv",
    mime="text/csv"
)

# -------------------------
# 1. Load Data
# -------------------------
uploaded_file = st.file_uploader("Upload HR dataset (CSV)", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded.")
else:
    df = generate_synthetic_hr()
    st.info("ℹ️ Using synthetic demo dataset.")

if "Attrition" in df.columns:
    if df["Attrition"].dropna().astype(str).str.lower().isin(["0","1"]).all():
        df["Attrition"] = df["Attrition"].astype(int).map({0:"No",1:"Yes"})

st.write("### 👀 Data preview")
st.dataframe(df.head(10), use_container_width=True)

# -------------------------
# 2. Split Data
# -------------------------
target = "Attrition"
if target not in df.columns:
    st.error("❌ Dataset must have column 'Attrition' (Yes/No).")
    st.stop()

X = df.drop(columns=[target])
y = (df[target].astype(str).str.strip().str.lower()=="yes").astype(int)

cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object","category"]).columns.tolist()

preprocess = ColumnTransformer([
    ("cat", make_ohe(), cat_cols),
    ("num", "passthrough", num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y if y.nunique()==2 else None,
    test_size=0.25, random_state=42
)

# -------------------------
# 3. Train Model
# -------------------------
st.subheader("⚙️ Train Model")
model_choice = st.radio("Choose model:", ["Logistic Regression","Random Forest"], horizontal=True)

if model_choice=="Logistic Regression":
    model = Pipeline([("prep", preprocess), ("clf", LogisticRegression(max_iter=400))])
else:
    model = Pipeline([("prep", preprocess), ("clf", RandomForestClassifier(n_estimators=300, random_state=42))])

model.fit(X_train, y_train)
st.success(f"✅ Model trained: {model_choice}")

prep = model.named_steps["prep"]
clf = model.named_steps["clf"]

X_train_enc = to_dense_float(prep.transform(X_train))
X_test_enc = to_dense_float(prep.transform(X_test))
feature_names = prep.get_feature_names_out()
friendly_features = [friendly_name(f) for f in feature_names]

# -------------------------
# 4. SHAP Global
# -------------------------
st.subheader("🌍 Global Explanations (SHAP)")
st.caption("Which features have the strongest overall effect on attrition?")

top_feats = pd.DataFrame()
policy = []

try:
    if isinstance(clf, RandomForestClassifier):
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_enc)
        sv = np.array(shap_values[1])
    elif isinstance(clf, LogisticRegression):
        explainer = shap.LinearExplainer(clf, X_train_enc)
        sv = np.array(explainer.shap_values(X_test_enc))
        if sv.ndim==1:
            sv = sv.reshape(-1, X_test_enc.shape[1])
    else:
        sv=None

    if sv is not None:
        min_feat=min(sv.shape[1], X_test_enc.shape[1], len(friendly_features))
        sv,X_test_enc,friendly_features = sv[:,:min_feat],X_test_enc[:,:min_feat],friendly_features[:min_feat]

        fig,ax=plt.subplots()
        shap.summary_plot(sv,X_test_enc,feature_names=friendly_features,show=False,plot_type="bar")
        plt.tight_layout()
        st.pyplot(fig,clear_figure=True)

        global_mean_abs=np.abs(sv).mean(axis=0)
        feat_imp=pd.DataFrame({"feature":friendly_features,"mean_abs_shap":global_mean_abs})
        top_feats=feat_imp.sort_values("mean_abs_shap",ascending=False).head(10)
        st.dataframe(top_feats,use_container_width=True)

except Exception as e:
    st.error(f"SHAP failed: {e}")

# -------------------------
# 5. LIME Local (Pre-encoded stable)
# -------------------------
st.subheader("👤 Local Explanations (LIME)")
st.caption("Pick one employee to see why attrition was predicted.")

if len(X_test_enc)>0:
    sample_id=st.slider("Select employee index",0,len(X_test_enc)-1,0)

    def clf_predict(x):
        return clf.predict_proba(x)

    lime_explainer=lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_enc,
        feature_names=friendly_features,
        class_names=["No","Yes"],
        categorical_features=None,
        mode="classification",
        discretize_continuous=False
    )

    exp=lime_explainer.explain_instance(
        X_test_enc[sample_id],
        clf_predict,
        num_features=8
    )

    st.table(pd.DataFrame(exp.as_list(label=1),columns=["Factor","Effect"]))
else:
    st.info("Not enough rows for LIME.")

# -------------------------
# 6. Policy Levers
# -------------------------
st.subheader("🔧 Policy Levers")
LEVER_MAP={
    "OverTime":"Review overtime policy; offer compensatory offs.",
    "BusinessTravel":"Rotate travel assignments; enable remote work.",
    "MonthlyIncome":"Audit salary bands; align with market.",
    "JobSatisfaction":"Manager coaching; recognition programs.",
    "YearsAtCompany":"Create mid-tenure growth pathways."
}
if not top_feats.empty:
    base_feats=[f.split(":")[0] for f in top_feats["feature"]]
    policy=[(f,LEVER_MAP.get(f,"Define HR intervention.")) for f in base_feats[:5]]
    st.table(pd.DataFrame(policy,columns=["Driver","Suggested Action"]))

# -------------------------
# 7. Export Report
# -------------------------
st.subheader("📥 Export Managerial Report")
if st.button("Generate & download HTML report"):
    html=f"""
    <html><body>
    <h2>XAI HR Report</h2>
    <p><b>Model:</b> {model_choice}</p>
    <h3>Top Global Drivers</h3>
    {top_feats.to_html(index=False) if not top_feats.empty else '<p>Not available</p>'}
    <h3>Policy Levers</h3>
    <ul>{''.join([f"<li>{f}: {a}</li>" for f,a in policy]) if policy else 'Not available'}</ul>
    <p><i>Generated by Streamlit app</i></p>
    </body></html>
    """
    st.download_button("Download HTML report",data=html,file_name="xai_hr_report.html",mime="text/html")

# -------------------------
# 8. Footer
# -------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px;">
        Developed by <b>Prof. Dinesh K.</b>
        <a href="https://linktr.ee/realdrdj" target="_blank">(link)</a>
    </div>
    """,
    unsafe_allow_html=True
)
