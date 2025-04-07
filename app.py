import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from bs4 import BeautifulSoup
import time
# Load the trained model and encoder
goal_model = joblib.load("goal_predictions.pkl")
result_model = joblib.load("result_predictions.pkl")
label_encoder = joblib.load("label_encoders.pkl")
label_encoded_columns = ["HomeTeam", "AwayTeam"]

# Streamlit UI
st.title("⚽ Match Prediction Dashboard")
DEFAULT_CSV = "live.csv"  # Ensure this file is in the same directory as app.py


st.write("View predictions of English Premier League Football Matches")
def scrape_season(season: str):
    all_matches = []
    global team_form
    team_form = {}

    for matchday in range(1, 39):
        print(f"Scraping {season} - Matchday {matchday}...")
        url = f"https://www.worldfootball.net/schedule/eng-premier-league-{season}-spieltag/{matchday}/"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table", {"class": "standard_tabelle"})
        if not table:
            continue

        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            if len(cols) < 5:
                continue

            date = cols[0].text.strip()
            home = cols[1].text.strip()
            score = cols[2].text.strip()
            away = cols[3].text.strip()

            # Convert team names
            home = team_name_mapping.get(home, home)
            away = team_name_mapping.get(away, away)

            if "-" not in score:
                continue
            home_goals, away_goals = map(int, score.split("-"))

            match_link = cols[2].find("a")["href"] if cols[2].find("a") else None
            match_stats = scrape_match_stats(match_link) if match_link else {}

            h_form = team_form.get(home, [])[-5:]
            a_form = team_form.get(away, [])[-5:]

            match_data = {
                "Season": season,
                "Matchday": matchday,
                "Date": date,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "HTFormPts": compute_form_points(h_form),
                "ATFormPts": compute_form_points(a_form),
                "HTFormPtsStr": "".join(h_form),
                "ATFormPtsStr": "".join(a_form),
                "HomeValue": team_to_value.get(home),
                "AwayValue": team_to_value.get(away)
            }

            for i in range(5):
                match_data[f"HM{i+1}"] = h_form[-(i+1)] if len(h_form) >= i+1 else None
                match_data[f"AM{i+1}"] = a_form[-(i+1)] if len(a_form) >= i+1 else None

            match_data.update(match_stats)
            all_matches.append(match_data)

            team_form.setdefault(home, []).append(result_letter(home_goals, away_goals))
            team_form.setdefault(away, []).append(result_letter(away_goals, home_goals))

    return all_matches




# Ensure the dataset matches model input features
model_features = joblib.load("model_features.pkl")  # Store feature names used during training

all_teams = np.unique(list(label_encoder["HomeTeam"].classes_) + [home_team, away_team])
label_encoder["HomeTeam"].classes_ = all_teams
label_encoder["AwayTeam"].classes_ = all_teams

team_mapping = {team: i for i, team in enumerate(label_encoder["HomeTeam"].classes_)}
home_encoded = team_mapping.get(home_team, -1)  # Default to -1 if unseen
away_encoded = team_mapping.get(away_team, -1)


for col in ["HomeTeam", "AwayTeam"]:  
    if col in match_data.columns:
        match_data[col] = match_data[col].astype("category").cat.codes

missing_cols = set(model_features) - set(match_data.columns)
extra_cols = set(match_data.columns) - set(model_features)

# Fix missing columns by adding them with default values (e.g., 0)
for col in missing_cols:
    match_data[col] = 0  # Default value, change if needed

# Remove extra columns
match_data = match_data[model_features]
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
