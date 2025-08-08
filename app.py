import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load model dan scaler
@st.cache_resource
def load_lstm_model():
    model = load_model("my_attention_lstm_model.keras")
    return model

@st.cache_data
def load_scaler():
    with open("scaler (7).pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

model = load_lstm_model()
scaler = load_scaler()

# Judul
st.title("🧠 Seq2Seq LSTM Forecasting - 60 langkah ke depan")

# Upload file CSV
uploaded_file = st.file_uploader("📤 Unggah file CSV berisi kolom 'tag_value'", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'tag_value' not in df.columns:
        st.error("⚠️ Kolom 'tag_value' tidak ditemukan dalam file.")
    elif len(df) < 60:
        st.warning("⚠️ Data minimal 60 baris untuk prediksi.")
    else:
        # Ambil 60 data terakhir
        data_input = df['tag_value'].values[-60:]
        last_index = df.index[-1] if df.index.name else df.shape[0] - 1

        # Normalisasi
        data_input_scaled = scaler.transform(data_input.reshape(-1, 1))
        encoder_input = data_input_scaled.reshape(1, 60, 1)

        # Decoder input: array nol shape (1, 60, 1)
        decoder_input = np.zeros((1, 60, 1))

        # Prediksi
        prediction_scaled = model.predict([encoder_input, decoder_input])
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()

        # Buat hasil prediksi
        forecast_index = range(last_index + 1, last_index + 61)
        result_df = pd.DataFrame({
            'step': forecast_index,
            'predicted_tag_value': prediction
        })

        # Tampilkan grafik
        st.subheader("📈 Hasil Prediksi 60 Langkah ke Depan")
        st.line_chart(result_df.set_index('step'))

        with st.expander("📄 Lihat data prediksi"):
            st.dataframe(result_df)

        # Tombol download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Hasil sebagai CSV", csv, "prediksi.csv", "text/csv")
else:
    st.info("Silakan unggah file CSV terlebih dahulu.")
