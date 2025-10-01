
# Explainable AI for HR Decisions (Streamlit App)

This Streamlit app demonstrates **explainable AI** (SHAP + LIME) for HR datasets,
such as employee attrition, promotion, or performance.

## Features
- Upload or use synthetic HR dataset
- Train Logistic Regression or Random Forest
- Global explanations with SHAP
- Local explanations with LIME
- Suggested HR policy levers based on top drivers
- Exportable HTML managerial report

## How to run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Deployment
1. Push to GitHub with `streamlit_app.py` and `requirements.txt`.
2. Connect repo to [Streamlit Community Cloud](https://share.streamlit.io).
3. Your app will be live at a URL like:
   ```
   https://your-username-your-repo.streamlit.app/
   ```

---
© 2025. Prepared for academic and professional demos by Prof. Dinesh K.
