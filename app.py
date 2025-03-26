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
DEFAULT_CSV = "live.csv"  # Ensure this file is in the same directory as app.py


st.write("View predictions of English Premier League Football Matches")

data = pd.read_csv(DEFAULT_CSV)
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



teams = sorted(label_encoder["HomeTeam"].classes_)  # Get team names from LabelEncoder

# Sidebar Inputs
st.sidebar.header("🔍 Select Match Teams")
home_team = st.sidebar.selectbox("🏠 Home Team", teams)
away_team = st.sidebar.selectbox("✈️ Away Team", teams)

# Convert selected teams into encoded values
home_encoded = label_encoder["HomeTeam"].transform([home_team])[0]
away_encoded = label_encoder["AwayTeam"].transform([away_team])[0]

# Create a DataFrame for prediction
match_data = pd.DataFrame({
    "HomeTeam": [home_encoded],
    "AwayTeam": [away_encoded],
    "Year": [2025]  # Default future year
})

match_data = match_data[model_features]
dmatrix_match = xgb.DMatrix(match_data, feature_names=model_features)

# Predict Goals
goal_predictions = goal_model.predict(dmatrix_match)
fthg, ftag = int(goal_predictions[0][0]), int(goal_predictions[0][1])

# Predict Result
result_prediction = result_model.predict(dmatrix_match)[0]
result_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
predicted_result = result_mapping.get(result_prediction, "Unknown")


# Predictions
st.subheader("⚽ Predicted Match Outcome")
st.write(f"🏠 **{home_team}** {fthg} - {ftag} **{away_team}** ✈️")
st.write(f"📊 **Predicted Result:** {predicted_result}")

st.sidebar.header("📊 Adjust Match Conditions")

# Adjust home & away team strength
home_advantage = st.sidebar.slider("🏠 Home Advantage (Boost Goals)", 0, 3, 0)
away_advantage = st.sidebar.slider("✈️ Away Advantage (Boost Goals)", 0, 3, 0)

# Update goal predictions
adjusted_fthg = fthg + home_advantage
adjusted_ftag = ftag + away_advantage
st.sidebar.write(f"🔹 Adjusted Score: {home_team} {adjusted_fthg} - {adjusted_ftag} {away_team}")


fig, ax = plt.subplots()
ax.bar(["Home Goals", "Away Goals"], [fthg, ftag], color=["blue", "red"])
ax.set_title("⚽ Predicted Goals")
ax.set_ylabel("Goals")

st.pyplot(fig)
