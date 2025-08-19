import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# --- Load saved pipeline ---
pipeline = joblib.load("protest_risk_pipeline.pkl")
model = pipeline.named_steps['classifier']
preprocessor = pipeline.named_steps['preprocessor']

# --- Identify columns from pipeline ---
categorical_cols = preprocessor.transformers_[0][2]
numeric_cols = preprocessor.transformers_[1][2]
all_features = list(categorical_cols) + list(numeric_cols)

# --- Streamlit page setup ---
st.set_page_config(page_title="Service Delivery Risk Predictor", layout="wide")
st.title("📊 Service Delivery Risk Predictor")

# --- Input method selection ---
input_method = st.radio("Select input method:", ("Upload CSV", "Manual Entry"))
predicted = False  # Flag to control feature display

# --- CSV Upload Option ---
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset loaded successfully!")
        st.dataframe(df.head())

        # Ensure all expected columns exist
        missing_cols = set(all_features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[all_features]

        # Predict on button click
        if st.button("Predict Risk"):
            df["Predicted_HighRisk"] = pipeline.predict(df)
            st.subheader("Predictions")
            st.dataframe(df[["Predicted_HighRisk"]].head())
            predicted = True

# --- Manual Entry Option ---
elif input_method == "Manual Entry":
    st.subheader("Enter feature values manually")
    manual_data = {}
    for col in all_features:
        if col in categorical_cols:
            options = preprocessor.named_transformers_['onehot'].categories_[categorical_cols.index(col)]
            manual_data[col] = st.selectbox(f"{col}", options)
        else:
            manual_data[col] = st.number_input(f"{col}", value=0)

    if st.button("Predict Risk"):
        df_manual = pd.DataFrame([manual_data])
        df_manual["Predicted_HighRisk"] = pipeline.predict(df_manual)
        st.subheader("Prediction Result")
        st.dataframe(df_manual)
        df = df_manual  # Reuse for feature importance
        predicted = True

# --- Display Top 5 Features after prediction ---
if predicted:
    st.subheader("Top 5 Features Contributing to High Risk")
    onehot_features = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([onehot_features, numeric_cols])
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": all_feature_names, "Importance": importances})\
                    .sort_values(by="Importance", ascending=False)
    
    # Table display
    st.table(importance_df.head(5))

    # Bar chart
    top5 = importance_df.head(5)
    fig, ax = plt.subplots()
    ax.barh(top5["Feature"], top5["Importance"], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 5 Risk Factors")
    st.pyplot(fig)

# --- Download Predictions ---
if predicted:
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv"
    )


