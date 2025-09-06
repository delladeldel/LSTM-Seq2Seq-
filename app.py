import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
from datetime import timedelta

# Load model dan scaler
encoder_model = load_model("seq2seq_encoder_model (1) (1).h5")
decoder_model = load_model("seq2seq_decoder_model (1) (1).h5")
scaler = joblib.load("scaler (9) (1).pkl")

input_len = 60
output_len = 60
n_features = 1

st.title("LSTM Seq2Seq (Encoder-Decoder) Forecasting")

uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'ddate' not in df.columns or 'tag_value' not in df.columns:
        st.error("File harus memiliki kolom 'ddate' dan 'tag_value'")
    else:
        df['ddate'] = pd.to_datetime(df['ddate'])
        df = df.sort_values('ddate')

        st.subheader("Preview Data")
        st.dataframe(df.tail(10))

        # Ambil 60 data terakhir
        data_input = df['tag_value'].values[-input_len:]
        last_ddate = df['ddate'].iloc[-1]

        # Normalisasi dan reshape
        data_input = scaler.transform(data_input.reshape(-1, 1))
        encoder_input = data_input.reshape(1, input_len, 1)

        # Encode input sequence
        state_h, state_c = encoder_model.predict(encoder_input)
        states = [state_h, state_c]

        # Decoder input awal (0)
        decoder_input = np.zeros((1, 1, 1))

        predictions_scaled = []

        for i in range(output_len):
            pred, h, c = decoder_model.predict([decoder_input] + states)
            pred_value = pred[0, 0, 0]
            predictions_scaled.append(pred_value)

            # Update decoder input dan state
            decoder_input = np.array(pred_value).reshape(1, 1, 1)
            states = [h, c]

        # Inverse transform hasil prediksi
        predictions = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1))

        # Buat rentang waktu prediksi
        time_interval = df['ddate'].diff().mode()[0] if df['ddate'].diff().mode().size > 0 else timedelta(seconds=10)
        future_dates = [last_ddate + (i + 1) * time_interval for i in range(output_len)]
        pred_df = pd.DataFrame({'ddate': future_dates, 'predicted_value': predictions.flatten()})

        # Plot
        st.subheader("Prediksi 60 Langkah ke Depan")
        fig, ax = plt.subplots()
        ax.plot(df['ddate'].iloc[-200:], df['tag_value'].iloc[-200:], label='Data Historis')
        ax.plot(pred_df['ddate'], pred_df['predicted_value'], label='Prediksi', color='red')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Prediksi data")
        st.dataframe(pred_df)
