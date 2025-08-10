import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load models ---
@st.cache_resource
def load_models():
    encoder_model = load_model("seq2seqencodermodel.keras")
    decoder_model = load_model("seq2seqdecodermodel.keras")
    return encoder_model, decoder_model

# --- Load scaler ---
@st.cache_resource
def load_scaler():
    return joblib.load("scaler.gz")

# --- Fungsi prediksi Seq2Seq ---
def predict_sequence(input_seq, encoder_model, decoder_model, output_steps):
    # Encode input sebagai state awal
    states_value = encoder_model.predict(input_seq)

    # Decoder input pertama: nol
    target_seq = np.zeros((1, 1, 1))

    output = []
    for _ in range(output_steps):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Simpan hasil
        output.append(output_tokens[0, 0, 0])

        # Update target_seq dengan output saat ini
        target_seq = np.array([[[output_tokens[0, 0, 0]]]])

        # Update states
        states_value = [h, c]

    return np.array(output)

# --- Streamlit UI ---
st.title("Seq2Seq LSTM Forecasting")
st.write("Upload CSV berisi data sensor dengan kolom `tag_value` untuk memprediksi nilai ke depan.")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    if 'tag_value' not in df.columns:
        st.error("CSV harus memiliki kolom `tag_value`.")
    else:
        st.write("Data Preview:", df.head())

        # Load model dan scaler
        encoder_model, decoder_model = load_models()
        scaler = load_scaler()

        # Parameter input/output
        input_steps = 60
        output_steps = 60

        # Ambil data terakhir sesuai input_steps
        data_input = df['tag_value'].values[-input_steps:]
        data_input = scaler.transform(data_input.reshape(-1, 1))
        data_input = data_input.reshape(1, input_steps, 1)

        # Prediksi
        pred_scaled = predict_sequence(data_input, encoder_model, decoder_model, output_steps)

        # Invers normalisasi
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        st.subheader("Hasil Prediksi")
        st.line_chart(pred)
        st.write(pred)
