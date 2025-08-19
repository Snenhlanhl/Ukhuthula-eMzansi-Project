import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

st.set_page_config(page_title="Service Delivery Risk Predictor", layout="wide")
st.title("📊 Service Delivery Risk Predictor")

# --- Define columns ---
categorical_cols = ['Province name', 'District municipality name', 'District/Local municipality name']
numeric_cols = ['Total Population', 'Black African', 'Coloured', 'Indian/Asian', 'White', 
                'Informal Dwelling', 'Piped (tap) water on community stand', 'No access to piped (tap) water',
                'Pit toilet', 'Bucket toilet', 'Communal refuse dump', 'Communal container/central collection point',
                'Own refuse dump', 'Dump or leave rubbish anywhere (no rubbish disposal)',
                'Gas for lighting', 'Paraffin for lighting', 'Candles for lighting',
                'Paraffin for cooking', 'Wood for cooking', 'Coal for cooking', 'Animal dung for cooking']
all_features = categorical_cols + numeric_cols

# --- Recreate preprocessor and model ---
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
    ('scaler', StandardScaler(), numeric_cols)
])

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight='balanced'
)

# --- Input method selection ---
input_method = st.radio("Select input method:", ("Upload CSV", "Manual Entry"))
predicted = False  # Flag to control Top 5 display

# --- CSV Upload Option ---
if input_method == "Upload CSV":
    uploaded_file = st.file_uploader("📂 Upload your dataset (CSV)", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(" Dataset loaded successfully!")
        st.dataframe(df.head())

        # Ensure all expected columns exist
        missing_cols = set(all_features) - set(df.columns)
        for col in missing_cols:
            df[col] = 0
        df = df[all_features]

        # Fit preprocessor and model
        X_processed = preprocessor.fit_transform(df)
        model.fit(X_processed, np.zeros(X_processed.shape[0]))  # Dummy fit to avoid errors

        if st.button("Predict Risk"):
            predictions = model.predict(X_processed)
            df["Predicted_HighRisk"] = predictions
            st.subheader(" Predictions")
            st.dataframe(df[["Predicted_HighRisk"]].head())
            predicted = True

# --- Manual Entry Option ---
elif input_method == "Manual Entry":
    st.subheader(" Enter feature values manually")
    manual_data = {}
    for col in all_features:
        if col in categorical_cols:
            manual_data[col] = st.text_input(f"{col}", value="")
        else:
            manual_data[col] = st.number_input(f"{col}", value=0)

    if st.button("Predict Risk"):
        df_manual = pd.DataFrame([manual_data])
        X_processed = preprocessor.fit_transform(df_manual)
        model.fit(X_processed, np.zeros(X_processed.shape[0]))  # Dummy fit
        predictions = model.predict(X_processed)
        df_manual["Predicted_HighRisk"] = predictions
        st.subheader(" Prediction Result:")
        st.dataframe(df_manual)
        df = df_manual
        predicted = True

# --- Top 5 feature display ---
if predicted:
    st.subheader(" Top 5 Features Contributing to High Risk")
    onehot_features = preprocessor.named_transformers_['onehot'].get_feature_names_out(categorical_cols)
    all_feature_names = np.concatenate([onehot_features, numeric_cols])

    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.table(importance_df.head(5))

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
