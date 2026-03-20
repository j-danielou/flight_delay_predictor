import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Flight Departure Risk Analyzer", page_icon="📊", layout="wide")

@st.cache_resource
def load_model():
    model = joblib.load('models/flight_xgboost_model.joblib')
    features = joblib.load('models/model_features.joblib')
    return model, features

model, model_columns = load_model()

with st.sidebar:
    st.header("About the Model")
    st.markdown("""
    This tool is powered by an **XGBoost** classifier trained on the *nycflights13* dataset.
    
    **Test Set Performance:**
    * F1-Score: 0.53
    * Recall: 65%
    
    *Note: Predictions are solely based on historical weather patterns and calendar correlations. Unforeseen operational issues and mechanical failures are not accounted for in this model.*
    """)

st.title("Flight Departure Risk Analyzer")
st.markdown("Enter the flight parameters and local departure weather to evaluate the probability of a delay.")

col1, col2 = st.columns(2)

with col1:
    st.header("📅 Flight Details")
    month = st.slider("Month of the Year", 1, 12, 8)
    
    days = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    day_of_week = st.selectbox("Day of the Week", options=list(days.keys()), format_func=lambda x: days[x])
    
    time_of_day = st.selectbox("Time of Day", ["Morning", "Afternoon", "Evening", "Night"])
    origin = st.selectbox("Origin Airport", ["JFK", "EWR", "LGA"])
    carrier = st.selectbox("Airline Carrier", ["DL", "UA", "AA", "B6", "EV", "MQ", "US", "WN"])

with col2:
    st.header("☁️ Weather at Departure")
    temp = st.number_input("Temperature (°F)", value=70.0, help="Fahrenheit")
    wind_speed = st.number_input("Wind Speed (mph)", value=15.0, help="Miles per hour")
    precip = st.number_input("Precipitation (inches)", value=0.0, help="Rain/Snow in inches")
    visib = st.number_input("Visibility (miles)", value=10.0, help="Visibility in miles (max 10)")

st.markdown("---")

if st.button("Predict Flight Status", use_container_width=True):
    
    ticket_data = {
        "month": month, "day_of_week": day_of_week, "time_of_day": time_of_day,
        "origin": origin, "carrier": carrier, "temp": temp, 
        "wind_speed": wind_speed, "precip": precip, "visib": visib
    }
    df_input = pd.DataFrame([ticket_data])
    
    df_encoded = pd.get_dummies(df_input)
    df_final = df_encoded.reindex(columns=model_columns, fill_value=0)
    
    with st.spinner("The model is analyzing the flight data..."):
        prediction = model.predict(df_final)[0]
        probabilite = model.predict_proba(df_final)[0][1]

        st.markdown("---")
        if prediction == 1:
            st.error("**High Risk of Delay!**")
            st.metric(label="Delay Probability", value=f"{probabilite * 100:.2f} %")
        else:
            st.success("**Flight is likely On Time**")
            st.metric(label="Delay Probability", value=f"{probabilite * 100:.2f} %")