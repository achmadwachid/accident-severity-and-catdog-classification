import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# --- CONFIG PAGE ---
st.set_page_config(
    page_title="Analisis Keparahan Kecelakaan & Deteksi Kucing/Anjing",
    page_icon="🚦",
    layout="wide"
)

# --- DIRECTORIES & CONSTANTS ---
MODEL_ACCIDENT_DIR = "./model_accident"
MODEL_CAT_DOG_PATH = "model_cnn_kucing_anjing.keras"
CLASS_INDICES_PATH = "class_indices.json"
MODEL_3CLASS_PATH = "model_cnn_3class_v3.keras"
CLASS_INDICES_3CLASS_PATH = "class_indices_3class_v3.json"
IMG_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.70

# --- CACHING ASSETS ---
@st.cache_resource
def load_cat_dog_assets():
    assets = {}
    if Path(MODEL_CAT_DOG_PATH).exists() and Path(CLASS_INDICES_PATH).exists():
        try:
            assets["model"] = tf.keras.models.load_model(MODEL_CAT_DOG_PATH)
            with open(CLASS_INDICES_PATH, "r") as f:
                class_indices = json.load(f)
            assets["class_names"] = {v: k for k, v in class_indices.items()}
            assets["ready"] = True
        except Exception as e:
            assets["ready"] = False
            assets["error"] = str(e)
    else:
        assets["ready"] = False
        assets["error"] = "File model (.keras) atau class_indices.json tidak ditemukan."
    return assets

@st.cache_resource
def load_3class_assets():
    assets = {}
    if Path(MODEL_3CLASS_PATH).exists() and Path(CLASS_INDICES_3CLASS_PATH).exists():
        try:
            assets["model"] = tf.keras.models.load_model(MODEL_3CLASS_PATH)
            with open(CLASS_INDICES_3CLASS_PATH, "r") as f:
                class_indices = json.load(f)
            assets["class_names"] = {v: k for k, v in class_indices.items()}
            assets["ready"] = True
        except Exception as e:
            assets["ready"] = False
            assets["error"] = str(e)
    else:
        assets["ready"] = False
        assets["error"] = "File model 3 class atau class_indices tidak ditemukan."
    return assets

@st.cache_resource
def load_accident_assets():
    assets = {}
    try:
        assets["model"] = joblib.load(os.path.join(MODEL_ACCIDENT_DIR, "model_xgb.joblib"))
        assets["scaler"] = joblib.load(os.path.join(MODEL_ACCIDENT_DIR, "scaler.joblib"))
        assets["oe"] = joblib.load(os.path.join(MODEL_ACCIDENT_DIR, "oe.joblib"))
        assets["label_encoders"] = joblib.load(os.path.join(MODEL_ACCIDENT_DIR, "label_encoders.joblib"))
        assets["frequency_maps"] = joblib.load(os.path.join(MODEL_ACCIDENT_DIR, "frequency_maps.joblib"))
        assets["feature_names"] = joblib.load(os.path.join(MODEL_ACCIDENT_DIR, "feature_names.joblib"))
        
        # Load medians & outlier limits if they exist (for matching notebook cleaning pipeline)
        medians_path = os.path.join(MODEL_ACCIDENT_DIR, "medians.joblib")
        outlier_path = os.path.join(MODEL_ACCIDENT_DIR, "outlier_limits.joblib")
        if os.path.exists(medians_path) and os.path.exists(outlier_path):
            assets["medians"] = joblib.load(medians_path)
            assets["outlier_limits"] = joblib.load(outlier_path)
        else:
            assets["medians"] = {}
            assets["outlier_limits"] = {}
        assets["ready"] = True
    except Exception as e:
        assets["ready"] = False
        assets["error"] = str(e)
    return assets

# Load assets
cat_dog_assets = load_cat_dog_assets()
accident_assets = load_accident_assets()
assets_3class = load_3class_assets()

# --- HELPER FUNCTIONS ---
def predict_image(model, image, class_names):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prob = model.predict(img_array, verbose=0)[0][0]
    predicted_idx = 1 if prob > 0.5 else 0
    predicted_label = class_names[predicted_idx]
    confidence = prob if predicted_idx == 1 else 1 - prob

    return predicted_label, float(confidence)

def predict_image_3class(model, image, class_names):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    probs = model.predict(img_array, verbose=0)[0]
    predicted_idx = int(np.argmax(probs))
    predicted_label = class_names[predicted_idx]
    confidence = probs[predicted_idx]

    return predicted_label, float(confidence)


# --- MAIN HEADER ---
st.title("🚦 Analisis Keparahan Kecelakaan & Deteksi Kucing/Anjing")
st.write("Sistem terintegrasi untuk mendeteksi tingkat keparahan kecelakaan dan mengidentifikasi rintangan hewan liar (kucing/anjing) demi menghindari tabrakan.")

# Expander Status Model agar tidak memadati layar
with st.expander("🛠️ Status Koneksi Model (Klik untuk detail)"):
    col_status1, col_status2, col_status3 = st.columns(3)
    with col_status1:
        if cat_dog_assets["ready"]:
            st.success("🐾 **Model Kucing & Anjing:** Ready ✅")
        else:
            st.error("🐾 **Model Kucing & Anjing:** Offline ❌")
            st.caption(f"Error: {cat_dog_assets.get('error')}")
    with col_status2:
        if accident_assets["ready"]:
            st.success("🚗 **Model Keparahan Kecelakaan:** Ready ✅")
        else:
            st.error("🚗 **Model Keparahan Kecelakaan:** Offline ❌")
            st.caption(f"Error: {accident_assets.get('error')}")
    with col_status3:
        if assets_3class["ready"]:
            st.success("🐾 **Model 3 Kelas (Anjing/Kucing/Lain):** Ready ✅")
        else:
            st.error("🐾 **Model 3 Kelas (Anjing/Kucing/Lain):** Offline ❌")
            st.caption(f"Error: {assets_3class.get('error')}")

if not cat_dog_assets["ready"] or not accident_assets["ready"] or not assets_3class["ready"]:
    st.warning("⚠️ Menunggu model dan preprocessor siap. Pastikan Anda telah mengekspor data dari Jupyter Notebook.")
    st.stop()

# Inisialisasi session state untuk menyimpan hasil prediksi kecelakaan
if "sev_pred" not in st.session_state:
    st.session_state.sev_pred = None
if "sev_probs" not in st.session_state:
    st.session_state.sev_probs = None

# ==========================================
# 1️⃣ LANGKAH 1: PREDIKSI KEPARAHAN KECELAKAAN
# ==========================================
st.markdown("---")
st.subheader("1️⃣ Prediksi Keparahan Dampak Kecelakaan")
st.write("Masukkan kondisi jalan dan cuaca di tempat kejadian perkara (TKP):")

col1, col2, col3 = st.columns(3)
with col1:
    weather_conds = sorted(accident_assets["label_encoders"]["Weather_Condition"].classes_)
    weather_cond = st.selectbox("Kondisi Cuaca", weather_conds, index=weather_conds.index("Clear") if "Clear" in weather_conds else 0)
    visibility = st.number_input("Jarak Pandang (Visibility miles)", value=10.0, min_value=0.0)
    hour = st.slider("Jam Kejadian (Hour)", 0, 23, 12)
    sunrise_sunset = st.selectbox("Waktu Hari", ["Day", "Night"], index=0)
    
with col2:
    states = sorted(accident_assets["label_encoders"]["State"].classes_)
    state = st.selectbox("Negara Bagian (State)", states, index=states.index("NY") if "NY" in states else 0)
    cities = sorted(list(accident_assets["frequency_maps"]["City"].keys()))
    city = st.selectbox("Kota (City)", cities, index=cities.index("New York") if "New York" in cities else 0)
    counties = sorted(list(accident_assets["frequency_maps"]["County"].keys()))
    county = st.selectbox("Wilayah (County)", counties, index=0)
    timezone = st.selectbox("Timezone", sorted(accident_assets["label_encoders"]["Timezone"].classes_), index=0)

with col3:
    distance = st.number_input("Jarak Kemacetan Estimasi (Distance miles)", value=0.2, min_value=0.0)
    start_lat = st.number_input("Latitude", value=40.7128, format="%.6f")
    start_lng = st.number_input("Longitude", value=-74.0060, format="%.6f")
    
    # Fitur jalan penting
    traffic_signal = st.checkbox("Ada Traffic Light (Rambu Lampu Merah)")
    crossing = st.checkbox("Ada Penyeberangan Jalan (Crossing)")
    stop_sign = st.checkbox("Ada Rambu STOP")

if st.button("🔮 Prediksi Keparahan Kecelakaan", type="primary", use_container_width=True):
    with st.spinner("Menghitung tingkat keparahan..."):
        # 1. Mulai dengan seluruh default median dari training set
        input_data = {}
        medians = accident_assets.get("medians", {})
        for feat in accident_assets["feature_names"]:
            input_data[feat] = medians.get(feat, 0.0)
            
        # 2. Overwrite dengan input aktual dari UI Streamlit
        input_data["Start_Lat"] = start_lat
        input_data["Start_Lng"] = start_lng
        input_data["Distance(mi)"] = distance
        input_data["Visibility(mi)"] = visibility
        input_data["Hour"] = float(hour)
        
        # Fitur binary
        input_data["Traffic_Signal"] = float(traffic_signal)
        input_data["Crossing"] = float(crossing)
        input_data["Stop"] = float(stop_sign)
        
        # Ordinal encoding untuk twilight
        twilight_input = pd.DataFrame([{
            "Sunrise_Sunset": sunrise_sunset,
            "Civil_Twilight": sunrise_sunset,
            "Nautical_Twilight": sunrise_sunset,
            "Astronomical_Twilight": sunrise_sunset
        }])
        twilight_encoded = accident_assets["oe"].transform(twilight_input)[0]
        
        input_data["Sunrise_Sunset"] = twilight_encoded[0]
        input_data["Civil_Twilight"] = twilight_encoded[1]
        input_data["Nautical_Twilight"] = twilight_encoded[2]
        input_data["Astronomical_Twilight"] = twilight_encoded[3]
        
        # Frequency encoding
        input_data["City"] = accident_assets["frequency_maps"]["City"].get(city, accident_assets["frequency_maps"]["City"]["Unknown"])
        input_data["County"] = accident_assets["frequency_maps"]["County"].get(county, accident_assets["frequency_maps"]["County"]["Unknown"])
        
        # Label encoding
        input_data["State"] = accident_assets["label_encoders"]["State"].transform([state])[0]
        input_data["Timezone"] = accident_assets["label_encoders"]["Timezone"].transform([timezone])[0]
        input_data["Weather_Condition"] = accident_assets["label_encoders"]["Weather_Condition"].transform([weather_cond])[0]
        
        # Handling wind direction jika dilatih di model
        if "Wind_Direction" in accident_assets["feature_names"]:
            default_wd = "CALM" if "CALM" in accident_assets["label_encoders"]["Wind_Direction"].classes_ else accident_assets["label_encoders"]["Wind_Direction"].classes_[0]
            input_data["Wind_Direction"] = accident_assets["label_encoders"]["Wind_Direction"].transform([default_wd])[0]
            
        # 3. Winsorization (Capping Outliers) dengan limit 1% dan 99% dari training set
        outlier_limits = accident_assets.get("outlier_limits", {})
        for col, (lower_limit, upper_limit) in outlier_limits.items():
            if col in input_data:
                input_data[col] = np.clip(input_data[col], lower_limit, upper_limit)
        
        # 4. Buat DataFrame sesuai urutan fitur model
        df_input = pd.DataFrame([input_data])
        df_input = df_input[accident_assets["feature_names"]]
        
        # 5. Scaling dan Prediksi
        X_scaled = accident_assets["scaler"].transform(df_input)
        df_scaled = pd.DataFrame(X_scaled, columns=accident_assets["feature_names"])
        
        pred_xgb = accident_assets["model"].predict(df_scaled)[0]
        st.session_state.sev_pred = int(pred_xgb + 1)
        st.session_state.sev_probs = accident_assets["model"].predict_proba(df_scaled)[0]

# Tampilkan Hasil Keparahan jika sudah diprediksi
if st.session_state.sev_pred is not None:
    st.write("### 📊 Hasil Keparahan Dampak Kecelakaan")
    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if st.session_state.sev_pred == 1:
            st.metric("Keparahan Risiko (Severity)", "Severity 1 (Rendah) 🟢")
            st.success("Kecelakaan berdampak minimal pada lalu lintas jalan.")
        elif st.session_state.sev_pred == 2:
            st.metric("Keparahan Risiko (Severity)", "Severity 2 (Sedang) 🟡")
            st.warning("Kecelakaan berdampak sedang, menyebabkan sedikit perlambatan.")
        elif st.session_state.sev_pred == 3:
            st.metric("Keparahan Risiko (Severity)", "Severity 3 (Cukup Tinggi) 🟠")
            st.error("Kecelakaan berisiko tinggi terhadap kelancaran arus jalan.")
        elif st.session_state.sev_pred == 4:
            st.metric("Keparahan Risiko (Severity)", "Severity 4 (Sangat Fatal) 🔴")
            st.error("Bahaya Kritis! Berisiko menutup seluruh jalan dan kemacetan fatal.")
    with col_res2:
        st.write("**Keyakinan Model (Probabilitas):**")
        for idx, prob in enumerate(st.session_state.sev_probs):
            st.write(f"Severity {idx+1}: {prob:.2%}")
            st.progress(float(prob))

# ==========================================
# 2️⃣ LANGKAH 2: IDENTIFIKASI RINTANGAN HEWAN (KUCING/ANJING)
# ==========================================
st.markdown("---")
st.subheader("2️⃣ Deteksi & Identifikasi Rintangan Hewan di Jalan")
st.write("Apakah ada rintangan berupa **kucing** atau **anjing** yang terdeteksi di jalan? Silakan unggah foto dari kamera dashboard/ponsel untuk mengidentifikasi.")

uploaded_file = st.file_uploader("Upload foto rintangan hewan (Kucing/Anjing) di jalan...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col_img1, col_img2 = st.columns([1, 2])
    with col_img1:
        st.image(image, caption="Foto Rintangan di Lintasan Jalan", use_container_width=True)
        
    with st.spinner("Mengidentifikasi jenis hewan..."):
        predicted_label, confidence = predict_image(cat_dog_assets["model"], image, cat_dog_assets["class_names"])
        
    with col_img2:
        if confidence >= CONFIDENCE_THRESHOLD:
            detected_animal = "Kucing 🐱" if predicted_label == "cat" else "Anjing 🐶"
            st.success(f"🚨 **Rintangan Teridentifikasi:** {detected_animal} (Keyakinan: {confidence:.2%})")
            
            # Integrasikan kedua informasi (Panduan Safety Driving & Pengereman)
            st.markdown("### 📋 Panduan Pengereman & Keselamatan Cerdas:")
            sev = st.session_state.sev_pred if st.session_state.sev_pred is not None else 2
            
            if sev >= 3:
                st.error(
                    f"⚠️ **PERINGATAN KRITIS (Keparahan Jalan Tinggi - Severity {sev}):**\n"
                    f"Kondisi jalan/lingkungan sangat berbahaya jika Anda melakukan pengereman mendadak demi menghindari **{detected_animal}**.\n"
                    "- JANGAN membanting setir secara mendadak ke kanan/kiri karena risiko mobil terbalik atau tabrakan beruntun sangat tinggi.\n"
                    "- Kurangi kecepatan secara bertahap dan nyalakan lampu hazard segera.\n"
                    "- Jika benturan tidak terhindarkan, prioritaskan stabilitas kendaraan untuk mencegah kecelakaan fatal bagi manusia."
                )
            else:
                st.warning(
                    f"🔔 **REKOMENDASI AMAN (Keparahan Jalan Rendah/Sedang - Severity {sev}):**\n"
                    f"Kondisi jalan/lingkungan cukup memadai untuk melakukan tindakan pencegahan.\n"
                    "- Anda dapat melakukan pengereman aman dan terkontrol untuk membiarkan **{detected_animal}** melintas.\n"
                    "- Selalu periksa kaca spion tengah dan samping sebelum memperlambat kendaraan secara mendadak."
                )
        else:
            st.warning("❓ Objek dalam gambar tidak teridentifikasi sebagai Kucing atau Anjing dengan tingkat keyakinan yang cukup.")

# ==========================================
# 3️⃣ LANGKAH 3: IDENTIFIKASI RINTANGAN HEWAN (3 KELAS)
# ==========================================
st.markdown("---")
st.subheader("3️⃣ Deteksi Rintangan Hewan (3 Kelas: Kucing/Anjing/Lainnya)")
st.write("Unggah foto dari kamera dashboard/ponsel untuk mengidentifikasi apakah objek adalah kucing, anjing, atau bukan keduanya (lainnya).")

uploaded_file_3c = st.file_uploader("Upload foto rintangan (3 Kelas)...", type=["jpg", "jpeg", "png"], key="upload_3class")

if uploaded_file_3c is not None:
    image_3c = Image.open(uploaded_file_3c)
    col_img1_3c, col_img2_3c = st.columns([1, 2])
    with col_img1_3c:
        st.image(image_3c, caption="Foto Rintangan (3 Kelas)", use_container_width=True)
        
    with st.spinner("Mengidentifikasi jenis objek (3 Kelas)..."):
        predicted_label_3c, confidence_3c = predict_image_3class(assets_3class["model"], image_3c, assets_3class["class_names"])
        
    with col_img2_3c:
        if confidence_3c >= CONFIDENCE_THRESHOLD:
            if predicted_label_3c == "cat":
                detected_animal_3c = "Kucing 🐱"
            elif predicted_label_3c == "dog":
                detected_animal_3c = "Anjing 🐶"
            else:
                detected_animal_3c = "Lainnya (Bukan Anjing/Kucing) 🛑"

            st.success(f"🚨 **Rintangan Teridentifikasi:** {detected_animal_3c} (Keyakinan: {confidence_3c:.2%})")
            
            # Integrasikan informasi Pengereman
            st.markdown("### 📋 Panduan Pengereman & Keselamatan Cerdas (3 Kelas):")
            sev_3c = st.session_state.sev_pred if st.session_state.sev_pred is not None else 2
            
            if predicted_label_3c in ["cat", "dog"]:
                if sev_3c >= 3:
                    st.error(
                        f"⚠️ **PERINGATAN KRITIS (Keparahan Jalan Tinggi - Severity {sev_3c}):**\n"
                        f"Kondisi jalan/lingkungan sangat berbahaya jika Anda melakukan pengereman mendadak demi menghindari **{detected_animal_3c}**.\n"
                        "- JANGAN membanting setir secara mendadak ke kanan/kiri karena risiko mobil terbalik atau tabrakan beruntun sangat tinggi.\n"
                        "- Kurangi kecepatan secara bertahap dan nyalakan lampu hazard segera.\n"
                        "- Jika benturan tidak terhindarkan, prioritaskan stabilitas kendaraan untuk mencegah kecelakaan fatal bagi manusia."
                    )
                else:
                    st.warning(
                        f"🔔 **REKOMENDASI AMAN (Keparahan Jalan Rendah/Sedang - Severity {sev_3c}):**\n"
                        f"Kondisi jalan/lingkungan cukup memadai untuk melakukan tindakan pencegahan.\n"
                        "- Anda dapat melakukan pengereman aman dan terkontrol untuk membiarkan **{detected_animal_3c}** melintas.\n"
                        "- Selalu periksa kaca spion tengah dan samping sebelum memperlambat kendaraan secara mendadak."
                    )
            else:
                st.info(
                    f"ℹ️ **OBJEK LAIN TERDETEKSI:**\n"
                    f"Objek teridentifikasi sebagai bukan Kucing atau Anjing.\n"
                    "- Tetap waspada, sesuaikan kecepatan, dan perhatikan kondisi sekitar sesuai dengan tingkat keparahan jalan (Severity {sev_3c})."
                )
        else:
            st.warning("❓ Objek dalam gambar tidak teridentifikasi dengan tingkat keyakinan yang cukup.")