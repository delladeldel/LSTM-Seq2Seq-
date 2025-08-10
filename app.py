import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
import joblib

# === 1. Load model dan scaler ===
encoder_model = load_model("seq2seqencodermodelfix.keras")
decoder_model = load_model("seq2seqdecodermodelfix.ker")
scaler = joblib.load("scaler (9).pkl")

# === 2. Parameter ===
input_len = 60   # jumlah data input
output_len = 60   # jumlah langkah yang diprediksi

# === 3. Upload file ===
st.title("Prediksi Seq2Seq LSTM")
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    # Baca data
    df = pd.read_csv(uploaded_file)
    
    if 'tag_value' not in df.columns:
        st.error("CSV harus punya kolom 'tag_value'")
    else:
        # Ambil data terakhir untuk input
        data_input = df['tag_value'].values[-input_len:]
        
        # Normalisasi
        data_input = scaler.transform(data_input.reshape(-1, 1))
        
        # Reshape untuk encoder
        encoder_input = data_input.reshape(1, input_len, 1)
        
        # Encode input
        state_h, state_c = encoder_model.predict(encoder_input)
        
        # Decoder start token (pakai nilai terakhir input)
        target_seq = np.array(data_input[-1]).reshape(1, 1, 1)
        
        predictions = []
        
        # Loop prediksi langkah demi langkah
        for _ in range(output_len):
            output_tokens, h, c = decoder_model.predict([target_seq, state_h, state_c])
            
            # Simpan hasil prediksi
            predictions.append(output_tokens[0, 0, 0])
            
            # Update target_seq jadi hasil prediksi terakhir
            target_seq = output_tokens.reshape(1, 1, 1)
            
            # Update state
            state_h, state_c = h, c
        
        # Balikkan ke skala asli
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Tampilkan hasil
        st.subheader("Hasil Prediksi 60 langkah ke depan")
        st.line_chart(predictions)
        st.write(predictions)
