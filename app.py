import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =======================
# Fungsi Load Model
# =======================
@st.cache_resource
def load_model(model_name):
    return joblib.load(model_name)

# =======================
# UI: Judul Aplikasi
# =======================
st.title("🛵 Prediksi Waktu Pengiriman Makanan")

st.markdown("""
Masukkan informasi pemesanan untuk memprediksi estimasi waktu pengiriman makanan menggunakan berbagai model machine learning.
""")

# =======================
# Sidebar: Pilih Model
# =======================
model_option = st.sidebar.selectbox(
    "Pilih Model Machine Learning",
    [
        "Linear Regression",
        "Decision Tree",
        "Random Forest",
        "Random Forest (Tuned)"
    ]
)

# =======================
# Input Data dari User
# =======================
st.subheader("📥 Input Data Pemesanan")

delivery_person_age = st.number_input("Umur Kurir", min_value=18, max_value=60, value=30)
delivery_person_ratings = st.slider("Rating Kurir", min_value=0.0, max_value=5.0, step=0.1, value=4.5)
restaurant_lat = st.number_input("Latitude Restoran", value=28.7041)
restaurant_long = st.number_input("Longitude Restoran", value=77.1025)
delivery_lat = st.number_input("Latitude Tujuan", value=28.5355)
delivery_long = st.number_input("Longitude Tujuan", value=77.3910)
vehicle_condition = st.slider("Kondisi Kendaraan (1-5)", min_value=1, max_value=5, value=3)
multiple_deliveries = st.selectbox("Jumlah Pengantaran", options=[0, 1, 2, 3], index=0)

# Kolom kategorikal
order_day = st.selectbox("Hari Pemesanan", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
road_traffic = st.selectbox("Kondisi Lalu Lintas", ["Low", "Medium", "High", "Jam"])
weather = st.selectbox("Kondisi Cuaca", ["Sunny", "Stormy", "Sandstorms", "Cloudy", "Fog"])
city = st.selectbox("Kota", ["Urban", "Semi-Urban", "Metropolitian"])
type_of_order = st.selectbox("Tipe Pesanan", ["Snack", "Drinks", "Meal", "Buffet"])
type_of_vehicle = st.selectbox("Tipe Kendaraan", ["motorcycle", "scooter", "electric_scooter", "bicycle"])

# =======================
# Preprocessing Input
# =======================
input_dict = {
    "Delivery_person_Age": delivery_person_age,
    "Delivery_person_Ratings": delivery_person_ratings,
    "Restaurant_latitude": restaurant_lat,
    "Restaurant_longitude": restaurant_long,
    "Delivery_location_latitude": delivery_lat,
    "Delivery_location_longitude": delivery_long,
    "Vehicle_condition": vehicle_condition,
    "multiple_deliveries": multiple_deliveries,
    "Day": order_day,
    "Road_traffic_density": road_traffic,
    "Weather_conditions": weather,
    "City": city,
    "Type_of_order": type_of_order,
    "Type_of_vehicle": type_of_vehicle
}

# Buat DataFrame dari input
input_df = pd.DataFrame([input_dict])

# One-hot encoding sesuai dengan model training (pastikan fitur sama!)
input_df_encoded = pd.get_dummies(input_df)
# Jika perlu, tambahkan kolom yang tidak ada agar cocok dengan model
expected_cols = joblib.load("model_columns.pkl")  # file yang berisi urutan kolom saat training
for col in expected_cols:
    if col not in input_df_encoded.columns:
        input_df_encoded[col] = 0
input_df_encoded = input_df_encoded[expected_cols]

# =======================
# Load Model
# =======================
model_files = {
    "Linear Regression": "linear_regression_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "Random Forest (Tuned)": "random_forest_tuned_model.pkl"
}

model = load_model(model_files[model_option])

# =======================
# Prediksi
# =======================
if st.button("🔍 Prediksi Waktu Pengiriman"):
    pred = model.predict(input_df_encoded)[0]
    st.success(f"🕒 Estimasi Waktu Pengiriman: {pred:.2f} menit")
