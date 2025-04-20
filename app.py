import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from bs4 import BeautifulSoup

# Load models and encoders
goal_model = joblib.load("goal_predictions.pkl")
result_model = joblib.load("result_predictions.pkl")
label_encoder = joblib.load("label_encoders.pkl")
model_features = joblib.load("model_features.pkl")

# Team name normalization
team_name_mapping = {
    'Man City': 'Manchester City',
    'Man Utd': 'Manchester United',
    'Leicester': 'Leicester City',
    'Spurs': 'Tottenham Hotspur',
    'Wolves': 'Wolverhampton Wanderers',
    'West Ham': 'West Ham United',
    'Newcastle': 'Newcastle United',
    'Brighton': 'Brighton & Hove Albion',
    'Norwich': 'Norwich City',
    'Sheffield Utd': 'Sheffield United',
    'Nottm Forest': 'Nottingham Forest',
    'Cardiff': 'Cardiff City',
    'Swansea': 'Swansea City',
    'Huddersfield': 'Huddersfield Town',
    'Hull': 'Hull City',
    'QPR': 'Queens Park Rangers',
    'Blackpool': 'Blackpool',
    'Bournemouth': 'AFC Bournemouth'
}

# Dummy squad values (can be replaced with actual Transfermarkt scraped values)
team_to_value = {team: np.random.randint(200, 1000) for team in team_name_mapping.values()}

# Helper functions
def result_letter(h, a):
    return 'W' if h > a else 'D' if h == a else 'L'

def compute_form_points(form_list):
    return sum([{'W': 3, 'D': 1, 'L': 0}[r] for r in form_list])

def scrape_matchday():
    url = "https://www.worldfootball.net/schedule/eng-premier-league/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "standard_tabelle"})
    matches = []

    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 5:
            continue

        date = cols[0].text.strip()
        home = team_name_mapping.get(cols[1].text.strip(), cols[1].text.strip())
        away = team_name_mapping.get(cols[3].text.strip(), cols[3].text.strip())

        matches.append({
            "Date": date,
            "HomeTeam": home,
            "AwayTeam": away
        })

    return matches

def predict_match(home, away):
    input_data = pd.DataFrame([{
        "HomeTeam": home,
        "AwayTeam": away,
        "HTFormPts": 0,
        "ATFormPts": 0,
        "HomeValue": team_to_value.get(home, 500),
        "AwayValue": team_to_value.get(away, 500)
    }])

    for col in ["HomeTeam", "AwayTeam"]:
        input_data[col] = input_data[col].apply(
            lambda x: label_encoder[col].transform([x])[0] if x in label_encoder[col].classes_ else -1
        )

    for col in model_features:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[model_features]
    dmatrix = xgb.DMatrix(input_data, feature_names=model_features)
    fthg, ftag = goal_model.predict(dmatrix)[0]
    result_pred = result_model.predict(dmatrix)[0]
    result_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    return int(fthg), int(ftag), result_map[result_pred]

# Streamlit UI
st.title("\ud83c\udfc0 Real-Time EPL Match Predictions")

matches = scrape_matchday()
for match in matches:
    fthg, ftag, result = predict_match(match["HomeTeam"], match["AwayTeam"])
    st.subheader(f"{match['HomeTeam']} vs {match['AwayTeam']}")
    st.write(f"Predicted Score: {fthg} - {ftag}")
    st.write(f"Predicted Result: {result}")


