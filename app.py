import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM
import time

# ----------------------
# Load model & scaler
# ----------------------
@st.cache_resource
def load_lstm_model(path="my_attention_lstm_model.keras"):
    model = load_model(path, compile=False)
    return model

@st.cache_data
def load_scaler(path="scaler (7).pkl"):
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    return scaler

model = load_lstm_model()
scaler = load_scaler()

# Optional: tampilkan summary model di expander agar bisa dicek (berguna untuk debug)
with st.expander("🔧 Model summary (klik untuk melihat)"):
    import io, sys
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    st.text(stream.getvalue())

# ----------------------
# UI
# ----------------------
st.title("🧠 Seq2Seq LSTM + Attention — Multi-step Autoregressive Forecast (60 steps)")

uploaded_file = st.file_uploader("📤 Unggah CSV berisi kolom 'tag_value'", type=["csv"])
window_size = st.number_input("Window size (timesteps/history)", min_value=1, value=60, step=1)
n_steps = st.number_input("Jumlah langkah prediksi", min_value=1, value=60, step=1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'tag_value' not in df.columns:
        st.error("⚠️ Kolom 'tag_value' tidak ditemukan.")
    elif len(df) < window_size:
        st.warning(f"⚠️ Data minimal {window_size} baris untuk prediksi.")
    else:
        st.success("✅ Data terupload — memproses...")

        # Ambil last window_size data
        raw_history = df['tag_value'].values[-window_size:]
        # Simpan last index (untuk label x-axis)
        last_index = df.index[-1] if df.index.name else df.shape[0] - 1

        # 1) Scale history with training scaler (very important)
        history_scaled = scaler.transform(raw_history.reshape(-1, 1)).reshape(window_size, 1)

        # 2) Build encoder_input shape (1, timesteps, features)
        encoder_input = history_scaled.reshape(1, window_size, 1)

        # 3) Initialize decoder_input: use history (scaled) — not zeros
        # We'll maintain decoder_input as shape (1, window_size, 1) and shift+append predictions each step.
        decoder_input = history_scaled.reshape(1, window_size, 1).astype(np.float32)

        # 4) Multi-step autoregressive loop (call full model each step)
        scaled_predictions = []
        t0 = time.time()
        for step in range(int(n_steps)):
            # model expects [encoder_input, decoder_input]
            # depending on how model was saved it might be model([encoder_input, decoder_input])
            pred_seq = model.predict([encoder_input, decoder_input], verbose=0)  # shape (1, window_size, 1) or (1, n_steps, features)
            # We assume model returns same length as decoder_input (common in Seq2Seq training)
            # Take first time-step of the returned sequence as the "next-step" prediction
            # Support both shapes: (1, window_size, 1) or (1, n_steps, features)
            if pred_seq.ndim == 3:
                next_pred_scaled = pred_seq[0, 0, 0]  # first timestep
            elif pred_seq.ndim == 2:
                next_pred_scaled = pred_seq[0, 0]     # fallback
            else:
                # unexpected shape — convert to 1D
                next_pred_scaled = np.ravel(pred_seq)[0]

            scaled_predictions.append(next_pred_scaled)

            # Shift decoder_input left by one timestep and append the new prediction at the end
            # decoder_input shape: (1, window_size, 1)
            decoder_input = np.roll(decoder_input, -1, axis=1)           # shift left
            decoder_input[0, -1, 0] = next_pred_scaled                  # append predicted scaled value

        elapsed = time.time() - t0

        # 5) Inverse scale predictions
        preds_arr = np.array(scaled_predictions).reshape(-1, 1)
        preds_inversed = scaler.inverse_transform(preds_arr).flatten()

        # 6) Build results DataFrame
        forecast_index = list(range(last_index + 1, last_index + 1 + int(n_steps)))
        result_df = pd.DataFrame({
            "step": forecast_index,
            "predicted_tag_value": preds_inversed
        })

        # 7) Display
        st.subheader("📈 Hasil Prediksi")
        st.markdown(f"Prediksi {n_steps} langkah selesai dalam {elapsed:.2f}s")
        st.line_chart(result_df.set_index("step"))
        with st.expander("📄 Tabel hasil prediksi"):
            st.dataframe(result_df)

        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download hasil CSV", csv, "prediksi.csv", "text/csv")

else:
    st.info("Silakan unggah file CSV untuk mulai memprediksi.")
