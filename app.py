import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tempfile
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Seq2Seq LSTM Forecasting")

st.title("📈 Seq2Seq LSTM — Upload & Predict (Auto)")

# --- Sidebar: parameters & model uploads ---
st.sidebar.header("Settings & Model files")

input_len = st.sidebar.number_input("Input sequence length (timesteps)", min_value=1, value=60, step=1)
output_len = st.sidebar.number_input("Output horizon (timesteps to predict)", min_value=1, value=60, step=1)
n_features = st.sidebar.number_input("Number of features", min_value=1, value=1, step=1)

st.sidebar.markdown("---")
st.sidebar.write("Upload model files (optional). If not uploaded, app will try to load from working dir.")
encoder_file = st.sidebar.file_uploader("Encoder model (.keras/.h5)", type=["keras","h5"], key="enc", help="Optional. If not provided, app loads 'seq2seqencodermodelfix.keras' from working dir.")
decoder_file = st.sidebar.file_uploader("Decoder model (.keras/.h5)", type=["keras","h5"], key="dec")
scaler_file  = st.sidebar.file_uploader("Scaler (pickle .pkl)", type=["pkl","pickle"], key="scaler")

# helper: save uploaded file to temp file and return path
def _save_uploaded(u_file, name_hint):
    if u_file is None:
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(u_file.name)[1], prefix=name_hint)
    tmp.write(u_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

# prepare model paths (uploaded files have priority)
enc_path = _save_uploaded(encoder_file, "encoder_") or "seq2seqencodermodelfix.keras"
dec_path = _save_uploaded(decoder_file, "decoder_") or "seq2seqdecodermodelfix.keras"
scaler_path = _save_uploaded(scaler_file, "scaler_") or "scaler (9).pkl"

st.sidebar.markdown(f"**Encoder path:** `{enc_path}`  \n**Decoder path:** `{dec_path}`  \n**Scaler path:** `{scaler_path}`")

# --- Load scaler ---
@st.cache_resource
def load_scaler(path):
    try:
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        return None

scaler = load_scaler(scaler_path)
if scaler is None:
    st.sidebar.error("Scaler not found or failed to load. Upload a valid scaler .pkl in the sidebar.")
    st.stop()

# --- Load models ---
@st.cache_resource
def load_models(enc_path, dec_path):
    try:
        enc = tf.keras.models.load_model(enc_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load encoder model from '{enc_path}': {e}")
    try:
        dec = tf.keras.models.load_model(dec_path, compile=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load decoder model from '{dec_path}': {e}")
    return enc, dec

try:
    encoder_model, decoder_model = load_models(enc_path, dec_path)
except Exception as e:
    st.error(str(e))
    st.stop()

st.sidebar.success("Models loaded (or uploaded) successfully ✅")

# --- File uploader for data ---
st.subheader("1) Upload CSV (with 'ddate' and 'tag_value' or at least 'tag_value')")
uploaded = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

if uploaded is None:
    st.info("Upload CSV file to start prediction.")
    st.stop()

# read CSV
try:
    df = pd.read_csv(uploaded)
except Exception as e:
    st.error(f"Cannot read CSV: {e}")
    st.stop()

# try to parse dates if 'ddate' present
if 'ddate' in df.columns:
    try:
        df['ddate'] = pd.to_datetime(df['ddate'])
    except Exception:
        # keep as is if parsing fails
        pass

# validate tag_value column
if 'tag_value' not in df.columns:
    st.error("CSV must contain column 'tag_value'.")
    st.stop()

st.write("Data preview (last 10 rows):")
st.dataframe(df.tail(10))

# Ensure enough rows (or pad)
series = df['tag_value'].values.astype(float)
if len(series) < input_len:
    st.warning(f"Uploaded series has length {len(series)} which is less than required input_len={input_len}. Padding with last value.")
    pad_len = input_len - len(series)
    if len(series) == 0:
        st.error("No data available to pad. Upload a CSV with at least 1 'tag_value'.")
        st.stop()
    pad_values = np.full((pad_len,), series[-1])
    series = np.concatenate([pad_values, series])

# take last input_len
last_seq = series[-input_len:].reshape(-1, 1)

# normalize
try:
    last_seq_scaled = scaler.transform(last_seq)
except Exception as e:
    st.error(f"Scaler.transform failed: {e}")
    st.stop()

# shape for encoder: (1, input_len, n_features)
encoder_input = last_seq_scaled.reshape((1, input_len, n_features))

# --- Predict function with robust shape checks ---
def predict_seq2seq(encoder_input, encoder_model, decoder_model, output_len, n_features):
    # 1) get encoder states
    enc_out = encoder_model.predict(encoder_input)
    # enc_out may be list/tuple [h, c] or single array; handle both
    if isinstance(enc_out, (list, tuple)) and len(enc_out) >= 2:
        state_h, state_c = enc_out[0], enc_out[1]
    elif isinstance(enc_out, np.ndarray):
        # sometimes a single array is returned - try to split in half
        if enc_out.shape[-1] % 2 == 0:
            half = enc_out.shape[-1] // 2
            state_h = enc_out[:, :half]
            state_c = enc_out[:, half:]
        else:
            raise ValueError("Encoder output shape ambiguous; expected two states or an even-sized single vector.")
    else:
        raise ValueError("Unexpected encoder output type.")

    # ensure shapes (batch, latent_dim)
    state_h = np.asarray(state_h)
    state_c = np.asarray(state_c)
    if state_h.ndim == 1:
        state_h = state_h.reshape(1, -1)
    if state_c.ndim == 1:
        state_c = state_c.reshape(1, -1)

    # 2) initial decoder input: zeros (batch, 1, n_features)
    target_seq = np.zeros((1, 1, n_features))

    outputs = []
    for _ in range(output_len):
        # decoder expects [target_seq, state_h, state_c] (common ordering)
        try:
            yhat, h, c = decoder_model.predict([target_seq, state_h, state_c])
        except Exception as e:
            # try alternate ordering: [target_seq] + list(enc_out) (some saved models expect list)
            try:
                yhat, h, c = decoder_model.predict([target_seq] + list(enc_out))
            except Exception as e2:
                raise RuntimeError(f"Decoder predict failed. Tried two common input orders. Errors:\n1) {e}\n2) {e2}")

        # yhat shape: (batch, 1, n_features) or (batch, timesteps, features)
        val = yhat[0, 0] if yhat.ndim >= 3 else yhat[0]
        outputs.append(val)

        # update for next iteration
        # set next input as previous prediction (reshape to (1,1,n_features))
        next_input = np.array(val).reshape(1, 1, -1)
        target_seq = next_input

        # update states
        state_h, state_c = h, c

    outputs = np.array(outputs).reshape(-1, n_features)
    return outputs

# --- Run prediction (with try/except to show useful messages) ---
try:
    scaled_preds = predict_seq2seq(encoder_input, encoder_model, decoder_model, output_len, n_features)
    # inverse scale
    try:
        preds = scaler.inverse_transform(scaled_preds)
    except Exception:
        # if scaler expects 2D, ensure shape
        preds = scaler.inverse_transform(scaled_preds.reshape(-1, 1)).flatten()
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- Create future timestamps if ddate exists, otherwise index ---
if 'ddate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ddate']):
    last_date = pd.to_datetime(df['ddate'].iloc[-1])
    # assume original data frequency same as previous interval; fallback to seconds=10
    if len(df) >= 2 and pd.api.types.is_datetime64_any_dtype(df['ddate']):
        freq = df['ddate'].diff().median()
        if pd.isnull(freq):
            delta = timedelta(seconds=10)
        else:
            delta = freq
    else:
        delta = timedelta(seconds=10)
    future_dates = [last_date + (i+1)*delta for i in range(output_len)]
    future_index = pd.to_datetime(future_dates)
else:
    future_index = np.arange(len(series), len(series) + output_len)

future_df = pd.DataFrame({
    "ddate": future_index,
    "pred_tag_value": preds.flatten()
})

st.subheader("Future predictions")
st.dataframe(future_df)

# plot historical last N and predicted
fig, ax = plt.subplots(figsize=(10, 4))
# plot last part of historical
hist_show = 200
hist_df = df[['ddate','tag_value']].copy() if 'ddate' in df.columns else pd.DataFrame({'tag_value': series})
if 'ddate' in hist_df.columns:
    ax.plot(hist_df['ddate'].tail(hist_show), hist_df['tag_value'].tail(hist_show), label='historical')
    ax.plot(future_df['ddate'], future_df['pred_tag_value'], 'rx-', label='prediction')
    ax.set_xlabel('ddate')
else:
    ax.plot(np.arange(len(series))[-hist_show:], series[-hist_show:], label='historical')
    ax.plot(np.arange(len(series), len(series)+output_len), future_df['pred_tag_value'], 'rx-', label='prediction')
ax.legend()
ax.grid(True)
plt.xticks(rotation=30)
st.pyplot(fig)

st.success("Prediction finished ✅")
