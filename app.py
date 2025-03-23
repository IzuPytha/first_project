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

# File uploader
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])
if uploaded_file is not None:
    try:
        # Read the file
        data = pd.read_csv(uploaded_file)
        st.write("### 📊 Preview of Uploaded Data:")
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


    except Exception as e:
        st.error(f"⚠️ An error occurred while processing the file: {e}")

else:
    st.warning("⚠️ Please upload a CSV file to proceed.")
