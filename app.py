import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
# Load the trained model and encoder
goal_model = joblib.load("goal_predictions.pkl")
result_model = joblib.load("result_predictions.pkl")
label_encoder = joblib.load("label_encoders.pkl")
label_encoded_columns = ["HomeTeam", "AwayTeam"]

# Streamlit UI
st.title("⚽ Match Prediction Dashboard")
st.write("Upload your CSV file and get predictions!")
DEFAULT_CSV = "live.csv"  # Ensure this file is in the same directory as app.py

# Streamlit UI
st.title("⚽ Match Prediction Dashboard")
st.write("View predictions with the default dataset or upload your own.")

# File uploader (optional)
uploaded_file = st.file_uploader("📂 Upload CSV file (optional)", type=["csv"])

# Use the uploaded file if provided, otherwise use the default dataset
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Using uploaded file.")
else:
    data = pd.read_csv(DEFAULT_CSV)
    st.info("ℹ️ Using default dataset.")

# Display data preview
st.write("### 📊 Preview of Data:")
st.dataframe(data.head())

# Identify categorical columns
categorical_columns = data.select_dtypes(exclude=["number"]).columns.tolist()

    # Apply Label Encoding (Ensuring 1D input)
for col, encoder in label_encoder.items():
    if col in data.columns:
        data[col] = data[col].apply(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

# Ensure the dataset matches model input features
model_features = joblib.load("model_features.pkl")  # Store feature names used during training
for col in model_features:
    if col not in data:
        data[col] = 0  # Add missing columns with default values

data = data[model_features]  # Align feature order
data = data.apply(pd.to_numeric, errors="coerce")
feature_names = list(data.columns)  


# Make predictions
dmatrix_data = xgb.DMatrix(data, feature_names=model_features)
goal_predictions = goal_model.predict(dmatrix_data)
result_predictions = result_model.predict(dmatrix_data)

# Assign predictions
data["Predicted_FTHG"] = goal_predictions[:, 0].round().astype(int)
data["Predicted_FTAG"] = goal_predictions[:, 1].round().astype(int)
data["Predicted_FTR"] = result_predictions  # Already classified

for col, encoder in label_encoder.items():
    if col in data.columns:
        reverse_mapping = {index: label for index, label in enumerate(encoder.classes_)}
        data[col] = data[col].map(reverse_mapping)



# Show results
st.write("### 🎯 Predictions:")
st.dataframe(data)

# Download button for results
csv = data.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")

