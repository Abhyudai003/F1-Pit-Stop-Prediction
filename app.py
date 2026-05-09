import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page config
st.set_page_config(
    page_title="F1 Pit Stop Predictor",
    page_icon="🏎️",
    layout="wide"
)

# Custom CSS for F1 aesthetic
st.markdown("""
<style>
    .main {background-color: #1a1a1a;}
    .stApp {background-color: #1a1a1a; color: #ffffff;}
    h1 {color: #e10600; font-family: 'Georgia', serif;}
    h2, h3 {color: #ffffff;}
    .stSelectbox label, .stNumberInput label, .stSlider label {color: #cccccc;}
    .prediction-high {background-color: #e10600; color: white; padding: 20px; 
                      border-radius: 10px; text-align: center; font-size: 24px;}
    .prediction-low {background-color: #00a651; color: white; padding: 20px; 
                     border-radius: 10px; text-align: center; font-size: 24px;}
</style>
""", unsafe_allow_html=True)

st.title("🏎️ F1 Pit Stop Predictor")
st.markdown("### Predict whether a car will pit on the next lap")
st.markdown("---")

# ─── Load models and encoders ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        with open('lgbm.pkl', 'rb') as f:
            lgbm = pickle.load(f)
        with open('cat.pkl', 'rb') as f:
            cat = pickle.load(f)
        with open('le_driver.pkl', 'rb') as f:
            le_driver = pickle.load(f)
        with open('le_compound.pkl', 'rb') as f:
            le_compound = pickle.load(f)
        with open('le_race.pkl', 'rb') as f:
            le_race = pickle.load(f)
        with open('driver_encoded.pkl', 'rb') as f:
            driver_pit_rate = pickle.load(f)
        with open('race_encoded.pkl', 'rb') as f:
            race_pit_rate = pickle.load(f)
        with open('compound_encoded.pkl', 'rb') as f:
            compound_pit_rate = pickle.load(f)
        return lgbm, cat, le_driver, le_compound, le_race, driver_pit_rate, race_pit_rate, compound_pit_rate
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please run the notebook to generate pkl files.")
        return None, None, None, None, None, None, None, None

lgbm, cat, le_driver, le_compound, le_race, driver_pit_rate, race_pit_rate, compound_pit_rate = load_models()

# ─── Input layout ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Driver & Car")
    
    # Driver list from encoder
    if le_driver is not None:
        driver_options = list(le_driver.classes_)
    else:
        driver_options = ["D001", "D002", "D003"]
    driver = st.selectbox("Driver ID", driver_options)
    
    compound = st.selectbox("Tyre Compound", ["HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"])
    
    if le_race is not None:
        race_options = list(le_race.classes_)
    else:
        race_options = ["Australian Grand Prix", "Bahrain Grand Prix"]
    race = st.selectbox("Race", race_options)
    
    year = st.selectbox("Year", [2022, 2023, 2024, 2025])

with col2:
    st.subheader("Lap Data")
    
    lap_number = st.number_input("Lap Number", min_value=1, max_value=78, value=25)
    stint = st.number_input("Stint Number", min_value=1, max_value=8, value=2)
    tyre_life = st.number_input("Tyre Life (laps)", min_value=1, max_value=77, value=15)
    position = st.number_input("Current Position", min_value=1, max_value=20, value=5)
    pit_stop = st.selectbox("Pit Stop This Lap?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col3:
    st.subheader("Performance Data")
    
    lap_time = st.number_input("Lap Time (seconds)", min_value=60.0, max_value=300.0, value=90.0, step=0.1)
    lap_time_delta = st.number_input("Lap Time Delta", min_value=-100.0, max_value=100.0, value=0.0, step=0.1)
    cumulative_deg = st.number_input("Cumulative Degradation", min_value=-300.0, max_value=300.0, value=-20.0, step=0.1)
    race_progress = st.number_input("Race Progress (0-1)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)
    position_change = st.number_input("Position Change", min_value=-18, max_value=18, value=0)

st.markdown("---")

# ─── Predict button ──────────────────────────────────────────────────────────
if st.button("🏁 Predict Pit Stop", use_container_width=True):
    
    if lgbm is None or cat is None:
        st.error("Models not loaded. Cannot make prediction.")
    else:
        # ── Feature engineering (must match training pipeline exactly) ──
        
        # Encode driver
        try:
            driver_enc = le_driver.transform([driver])[0]
        except ValueError:
            driver_enc = 0  # fallback for unseen driver
        
        # Encode compound
        try:
            compound_enc = le_compound.transform([compound])[0]
        except ValueError:
            compound_enc = 0
        
        # Encode race
        try:
            race_enc = le_race.transform([race])[0]
        except ValueError:
            race_enc = 0
        
        # Target encodings
        d_pit_rate = driver_pit_rate.get(driver, driver_pit_rate.mean())
        r_pit_rate = race_pit_rate.get(race, race_pit_rate.mean())
        c_pit_rate = compound_pit_rate.get(compound, compound_pit_rate.mean())
        
        # Engineered features
        degradation_by_stint = tyre_life / (stint + 1)
        lap_progress_feat = tyre_life / (lap_number + 1)
        tyre_life_lag1 = 0  # unknown for single prediction
        tyre_life_diff = 0  # unknown for single prediction
        laps_remaining = 0  # estimate — unknown without full race data
        
        # Build input row — column order must match X during training
        input_data = pd.DataFrame([{
            'Driver': driver_enc,
            'Compound': compound_enc,
            'Race': race_enc,
            'Year': year,
            'PitStop': pit_stop,
            'LapNumber': lap_number,
            'Stint': stint,
            'TyreLife': tyre_life,
            'Position': position,
            'LapTime (s)': lap_time,
            'LapTime_Delta': lap_time_delta,
            'Cumulative_Degradation': cumulative_deg,
            'RaceProgress': race_progress,
            'Position_Change': position_change,
            'DegradationByStint': degradation_by_stint,
            'LapProgress': lap_progress_feat,
            'TyreLife_lag1': tyre_life_lag1,
            'TyreLife_diff': tyre_life_diff,
            'driver_encoded': d_pit_rate,
            'race_encoded': r_pit_rate,
            'compound_encoded': c_pit_rate,
        }])
        
        # Ensemble prediction
        lgbm_prob = lgbm.predict_proba(input_data)[:, 1][0]
        cat_prob = cat.predict_proba(input_data)[:, 1][0]
        
        # Weighted ensemble matching your best submission
        final_prob = 0.50 * lgbm_prob + 0.50 * cat_prob
        
        # Display result
        st.markdown("### Prediction Result")
        
        if final_prob >= 0.5:
            st.markdown(f"""
            <div class="prediction-high">
                🔴 PIT STOP LIKELY<br>
                Probability: {final_prob:.1%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-low">
                🟢 CONTINUING ON TRACK<br>
                Pit Probability: {final_prob:.1%}
            </div>
            """, unsafe_allow_html=True)
        
        # Show breakdown
        st.markdown("#### Model Breakdown")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("LightGBM Probability", f"{lgbm_prob:.1%}")
        with col_b:
            st.metric("CatBoost Probability", f"{cat_prob:.1%}")

st.markdown("---")
st.markdown("*Built using LightGBM + CatBoost ensemble | Kaggle Playground Series S5*")