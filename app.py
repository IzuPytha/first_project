import requests
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
API_KEY = d8462708faa894faee68ffddf3ab31f2
API_URL = "https://api.football-data.org/v4/matches"

# Fetch live match data
def get_live_matches():
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(API_URL, headers=headers)
    
    if response.status_code == 200:
        matches = response.json().get("matches", [])
        match_list = [
            {"HomeTeam": match["homeTeam"]["name"], "AwayTeam": match["awayTeam"]["name"]}
            for match in matches
        ]
        return match_list
    else:
        st.error("⚠️ Unable to fetch match data.")
        return []

# Fetch upcoming matches
matches = get_live_matches()

# Team selection from live data
st.sidebar.header("🔍 Select a Live Match")
if matches:
    match_selection = st.sidebar.selectbox("Select Match", [f"{m['HomeTeam']} vs {m['AwayTeam']}" for m in matches])
    home_team, away_team = match_selection.split(" vs ")
else:
    home_team, away_team = "Man City", "Arsenal"  # Default teams

st.sidebar.write(f"🏠 **{home_team}** vs **{away_team}** ✈️")

def get_team_stats(team_name):
    team_url = f"https://api.football-data.org/v4/teams?name={team_name}"
    headers = {"X-Auth-Token": API_KEY}
    response = requests.get(team_url, headers=headers)
    
    if response.status_code == 200:
        team_data = response.json().get("teams", [{}])[0]
        return {
            "Form": team_data.get("lastFiveResults", "WDLWW"),
            "GoalsPerMatch": team_data.get("averageGoalsPerGame", 1.5)
        }
    return {"Form": "WDLDW", "GoalsPerMatch": 1.2}  # Default values

home_stats = get_team_stats(home_team)
away_stats = get_team_stats(away_team)

st.sidebar.write(f"🏠 {home_team} Form: {home_stats['Form']} | Goals: {home_stats['GoalsPerMatch']}")
st.sidebar.write(f"✈️ {away_team} Form: {away_stats['Form']} | Goals: {away_stats['GoalsPerMatch']}")



# Identify categorical columns

# Ensure the dataset matches model input features
model_features = joblib.load("model_features.pkl")  # Store feature names used during training


# Convert selected teams into encoded values
home_encoded = label_encoder["HomeTeam"].transform([home_team])[0]
away_encoded = label_encoder["AwayTeam"].transform([away_team])[0]

match_data = pd.DataFrame({
    "HomeTeam": [home_encoded],
    "AwayTeam": [away_encoded],
    "HTFormPtsStr": [home_stats["Form"]],
    "ATFormPtsStr": [away_stats["Form"]],
    "Year": [2025]
})

# Convert to DMatrix and predict
dmatrix_match = xgb.DMatrix(match_data, feature_names=model_features)
goal_predictions = goal_model.predict(dmatrix_match)
fthg, ftag = int(goal_predictions[0][0]), int(goal_predictions[0][1])

result_predictions = result_model.predict(dmatrix_match)[0]
result_mapping = {0: "Home Win", 1: "Draw", 2: "Away Win"}
predicted_result = result_mapping.get(result_predictions, "Unknown")

# Display Predictions
st.subheader("⚽ Predicted Match Outcome")
st.write(f"🏠 **{home_team}** {fthg} - {ftag} **{away_team}** ✈️")
st.write(f"📊 **Predicted Result:** {predicted_result}")
