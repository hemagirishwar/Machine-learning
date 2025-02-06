import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

st.title("Laptop Price Predictor")

# User inputs
company = st.selectbox("Brand", df["Company"].unique())
typee = st.selectbox("Type", df["TypeName"].unique())
ram = st.selectbox("Ram (in GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input("Enter Weight of the Laptop (in kg)", format="%.2f", step=0.01)
touch = st.selectbox("Touch Screen (1 = Yes, 0 = No)", [1, 0])
ips = st.selectbox("IPS (1 = Yes, 0 = No)", [1, 0])
screen_size = st.number_input("Screen Size (in inches)", format="%.1f", step=0.1)
resolution = st.selectbox(
    "Screen Resolution",
    ["1920x1080", "1366x768", "1600x900", "3840x2160", "3200x1800", "2880x1800", "2560x1600", "2560x1440", "2304x1440"],
)
cpu = st.selectbox("CPU Brand", df["CPU_brand"].unique())
hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox("GPU Brand", df["GPU_model"].unique())
os = st.selectbox("OS", df["OS"].unique())

# Predict button
if st.button("Predict Price"):
    try:
        # Process resolution for PPI calculation
        x_res, y_res = map(int, resolution.split("x"))
        ppi = ((x_res**2) + (y_res**2)) ** 0.5 / screen_size

        # Prepare query for prediction
        query = np.array([company, typee, ram, os , weight, touch, ips, gpu, cpu, ssd, hdd, ppi])
        query = query.reshape(1, -1)

        # Predict and display the price
        predicted_price = pipe.predict(query)[0]
        st.title(f"Predicted Price: {round(np.exp(predicted_price), 2)}")

    except ValueError as e:
        st.error(f"Error during prediction: {e}. Please check your inputs.")
