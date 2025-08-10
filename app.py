import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# ======== Load Model & Scaler ========
@st.cache_resource
def load_models():
    encoder_model = load_model("seq2seqencodermodelfix.keras")
    decoder_model = load_model("seq2seqdecodermodelfix.keras")
    return encoder_model, decoder_model

@st.cache_resource
def load_scaler():
    with open("scaler (9).pkl", "rb") as f:
        scaler = pickle.load(f)
    return scaler

encoder_model, decoder_model = load_models()
scaler = load_scaler()

# ======== Inference Function ========
def predict_seq(input_sequence, input_len=120, output_len=60):
    # Normalisasi input
    input_sequence = scaler.transform(np.array(input_sequence).reshape(-1, 1))
    input_sequence = input_sequence.reshape(1, input_len, 1)

    # Encode input sequence
    state_h, state_c = encoder_model.predict(input_sequence)

    # Decoder input awal (nol semua)
    decoder_input = np.zeros((1, 1, 1))

    decoded_output = []
    for _ in range(output_len):
        output, state_h, state_c = decoder_model.predict([decoder_input, state_h, state_c])
        decoded_output.append(output[0, 0, 0])
        decoder_input = output

    # Denormalisasi hasil
    decoded_output = scaler.inverse_transform(np.array(decoded_output).reshape(-1, 1)).flatten()
    return decoded_output

# ======== Streamlit UI ========
st.title("📈 Seq2Seq LSTM Forecasting")
st.write("Prediksi nilai sensor **tag_value** untuk 10 menit ke depan.")

uploaded_file = st.file_uploader("Upload file CSV data sensor (harus ada kolom 'tag_value')", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "tag_value" not in df.columns:
        st.error("CSV harus memiliki kolom 'tag_value'.")
    elif len(df) < 120:
        st.error("Data minimal harus memiliki 120 baris untuk prediksi.")
    else:
        st.success("File diterima!")
        st.line_chart(df["tag_value"], height=200)

        # Ambil 120 data terakhir
        last_data = df["tag_value"].values[-120:]

        # Prediksi
        prediction = predict_seq(last_data)

        st.subheader("Hasil Prediksi")
        st.line_chart(prediction, height=200)

        # Gabung data aktual + prediksi
        df_pred = pd.DataFrame({
            "Actual": np.concatenate([df["tag_value"].values, [np.nan]*60]),
            "Prediction": np.concatenate([[np.nan]*len(df), prediction])
        })

        st.subheader("Aktual vs Prediksi")
        st.line_chart(df_pred, height=300)

st.caption("Dibuat dengan ❤️ menggunakan LSTM Seq2Seq")
