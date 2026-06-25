import streamlit as st
import requests

st.title("Machine Failure Prediction System")

air_temp = st.number_input("Air Temperature [K]", value=300.0)
process_temp = st.number_input("Process Temperature [K]", value=310.0)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1500)
torque = st.number_input("Torque [Nm]", value=40.0)
tool_wear = st.number_input("Tool Wear [min]", value=10)

if st.button("Predict"):

    features = [
        air_temp,
        process_temp,
        rot_speed,
        torque,
        tool_wear
    ]

    url = "http://127.0.0.1:5000/predict"

    response = requests.post(
        url,
        json={"features": features}
    )

    st.write("Status Code:", response.status_code)
    st.write("Response Text:", response.text)

    try:
        result = response.json()

        st.success(f"Prediction: {result['prediction']}")
        st.info(f"Explanation: {result['explanation']}")

    except Exception as e:
        st.error(f"JSON Error: {e}")