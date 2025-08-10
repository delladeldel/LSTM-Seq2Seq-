import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

# ========================
# Load Model & Scaler
# ========================
encoder_model = tf.keras.models.load_model("seq2seqencodermodelfix.keras")
decoder_model = tf.keras.models.load_model("seq2seqdecodermodelfix.keras")

with open("scaler (9).pkl", "rb") as f:
    scaler = pickle.load(f)

# ========================
# Streamlit UI
# ========================
st.title("🔮 Seq2Seq Forecasting - 10 Menit ke Depan")
st.write("Upload file CSV dengan kolom **tag_value** untuk memprediksi 60 langkah ke depan.")

uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

if uploaded_file:
    # Baca data
    df = pd.read_csv(uploaded_file)

    if "tag_value" not in df.columns:
        st.error("CSV harus memiliki kolom 'tag_value'")
    else:
        st.success("Data berhasil dibaca!")

        # Ambil 60 data terakhir
        input_len = 60
        last_sequence = df["tag_value"].values[-input_len:]

        # Scaling
        last_sequence_scaled = scaler.transform(last_sequence.reshape(-1, 1))
        encoder_input = last_sequence_scaled.reshape(1, input_len, 1)

        # ========================
        # Inference
        # ========================
        # Encode input sequence
        state_h, state_c = encoder_model.predict(encoder_input)

        # Decoder prediction loop
        output_len = 60
        decoder_input = np.zeros((1, 1, 1))
        states = [state_h, state_c]
        outputs = []

        for _ in range(output_len):
            decoder_output, state_h, state_c = decoder_model.predict([decoder_input] + states)
            outputs.append(decoder_output[0, 0])
            decoder_input = decoder_output.reshape(1, 1, 1)
            states = [state_h, state_c]

        # Inverse scaling
        outputs_rescaled = scaler.inverse_transform(np.array(outputs).reshape(-1, 1)).flatten()

        # ========================
        # Tampilkan Hasil
        # ========================
        st.subheader("📈 Hasil Prediksi")
        fig, ax = plt.subplots()
        ax.plot(range(len(df)), df["tag_value"], label="Data Asli")
        ax.plot(range(len(df), len(df) + output_len), outputs_rescaled, label="Prediksi", color="red")
        ax.legend()
        st.pyplot(fig)

        st.write("**Hasil Prediksi (60 langkah):**")
        st.dataframe(pd.DataFrame({"Prediksi": outputs_rescaled}))
