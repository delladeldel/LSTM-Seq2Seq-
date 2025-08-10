import streamlit as st
import pandas as pd
import numpy as np
import pickle, joblib, tempfile, os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

st.set_page_config(page_title="Seq2Seq Forecast", layout="wide")
st.title("Seq2Seq LSTM — Upload & Predict (robust inference)")

# -------------------------
# Sidebar: upload files
# -------------------------
st.sidebar.header("Upload model & scaler (or keep files in repo)")
enc_file = st.sidebar.file_uploader("Encoder model (.keras/.h5) — optional", type=["keras","h5"])
dec_file = st.sidebar.file_uploader("Decoder model (.keras/.h5) — optional", type=["keras","h5"])
train_file = st.sidebar.file_uploader("Training model (seq2seq_training) (.keras/.h5) — optional", type=["keras","h5"])
scaler_file = st.sidebar.file_uploader("Scaler (pickle .pkl/.gz) — required if not in repo", type=["pkl","gz","pickle"])

# helper to write uploaded file to temp path
def _save_uploaded(u_file, name):
    if u_file is None: return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(u_file.name)[1], prefix=name)
    tmp.write(u_file.getbuffer())
    tmp.flush(); tmp.close()
    return tmp.name

enc_path = _save_uploaded(enc_file, "enc_") or "seq2seq_encoder_model (1).keras"
dec_path = _save_uploaded(dec_file, "dec_") or "seq2seq_decoder_model (1).keras"
train_path = _save_uploaded(train_file, "train_") or "seq2seq_training_model (1).keras"
scaler_path = _save_uploaded(scaler_file, "scaler_") or "scaler (9).pkl"

st.sidebar.write("Paths used:")
st.sidebar.code(f"encoder -> {enc_path}")
st.sidebar.code(f"decoder -> {dec_path}")
st.sidebar.code(f"training -> {train_path}")
st.sidebar.code(f"scaler -> {scaler_path}")

# -------------------------
# Load scaler (required)
# -------------------------
scaler = None
if os.path.exists(scaler_path):
    try:
        # try joblib then pickle
        try:
            scaler = joblib.load(scaler_path)
        except Exception:
            with open(scaler_path,"rb") as f:
                scaler = pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load scaler: {e}")
        st.stop()
else:
    st.warning("Scaler not found at path; please upload scaler.pkl in sidebar or place it in repo.")
    st.stop()

# -------------------------
# Function to load or build inference models
# -------------------------
@st.cache_resource
def get_models(enc_path, dec_path, train_path):
    enc_model = None
    dec_model = None
    training_model = None

    # Try loading encoder & decoder first
    if os.path.exists(enc_path) and os.path.exists(dec_path):
        try:
            enc_model = load_model(enc_path, compile=False)
            dec_model = load_model(dec_path, compile=False)
            return enc_model, dec_model, None
        except Exception as e:
            # fall through to try training model
            st.sidebar.write(f"Warning loading separate models: {e}")

    # Try loading training model and reconstruct inference models
    if os.path.exists(train_path):
        try:
            training_model = load_model(train_path, compile=False)
        except Exception as e:
            raise RuntimeError(f"Cannot load training model: {e}")

        # Attempt to reconstruct encoder & decoder from training_model
        # Heuristics: training_model.inputs contains encoder_inputs and decoder_inputs (2 inputs)
        try:
            # get input layers
            t_inputs = training_model.inputs
            t_outputs = training_model.outputs
            # assume encoder input is first input
            encoder_inputs = t_inputs[0]
            decoder_inputs = t_inputs[1]

            # find LSTM layers by name / type in model.layers
            # we'll re-call layers with new Input objects to create inference models
            # simpler approach: re-create a new encoder-decoder architecture with shapes taken from training model

            # deduce input_len and n_features from encoder_inputs
            enc_shape = tuple(encoder_inputs.shape.as_list())  # (None, timesteps, features)
            input_len = int(enc_shape[1]) if enc_shape[1] is not None else 60
            n_features = int(enc_shape[2]) if enc_shape[2] is not None else 1

            # deduce latent_dim by inspecting LSTM layer in model
            latent_dim = None
            lstm_layers = [ly for ly in training_model.layers if isinstance(ly, tf.keras.layers.LSTM)]
            if len(lstm_layers) >= 1:
                # assume first LSTM is encoder with units attr
                latent_dim = lstm_layers[0].units
            else:
                latent_dim = 64

            # Build new encoder inference model
            enc_inputs_new = Input(shape=(input_len, n_features), name="enc_infer_input")
            # find the encoder LSTM layer object from training_model by matching units if possible
            # we'll create a new LSTM layer with same units and then load weights from the appropriate layer in training_model
            # Find the encoder LSTM in training model by searching layers with return_state True or by name
            # As fallback use first LSTM layer
            enc_lstm = tf.keras.layers.LSTM(latent_dim, return_state=True, name="enc_lstm_infer")
            enc_out, state_h, state_c = enc_lstm(enc_inputs_new)
            encoder_model = Model(enc_inputs_new, [state_h, state_c])

            # Build new decoder inference model (single-step)
            dec_inputs_new = Input(shape=(1, n_features), name="dec_infer_input")  # one timestep input for inference
            dec_state_h = Input(shape=(latent_dim,), name="dec_state_h")
            dec_state_c = Input(shape=(latent_dim,), name="dec_state_c")
            dec_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="dec_lstm_infer")
            dec_outs, dec_h, dec_c = dec_lstm(dec_inputs_new, initial_state=[dec_state_h, dec_state_c])
            dec_dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features), name="dec_dense_infer")
            dec_outputs = dec_dense(dec_outs)
            decoder_model = Model([dec_inputs_new, dec_state_h, dec_state_c], [dec_outputs, dec_h, dec_c])

            # Now copy weights from training_model's encoder/decoder layers to our new layers where possible
            # Strategy: map by layer type & units. This is heuristic and may require layer name matching.
            # Build a mapping of training model LSTM and Dense layers
            train_lstm = [ly for ly in training_model.layers if isinstance(ly, tf.keras.layers.LSTM)]
            train_dense = [ly for ly in training_model.layers if isinstance(ly, tf.keras.layers.Dense)]
            # If we have at least 2 LSTM layers, map first->enc, second->dec
            if len(train_lstm) >= 2:
                # copy weights
                try:
                    encoder_lstm_layer = train_lstm[0]
                    decoder_lstm_layer = train_lstm[1]
                    encoder_lstm_weights = encoder_lstm_layer.get_weights()
                    decoder_lstm_weights = decoder_lstm_layer.get_weights()
                    enc_lstm.set_weights(encoder_lstm_weights)
                    dec_lstm.set_weights(decoder_lstm_weights)
                except Exception as e:
                    st.sidebar.write(f"Warning copying LSTM weights failed: {e}")

            # copy dense weights if available
            if len(train_dense) >= 1:
                try:
                    dec_dense.set_weights(train_dense[-1].get_weights())
                except Exception as e:
                    st.sidebar.write(f"Warning copying Dense weights failed: {e}")

            return encoder_model, decoder_model, training_model

        except Exception as e:
            raise RuntimeError(f"Failed to build inference models from training model: {e}")

    raise RuntimeError("No valid encoder/decoder or training model found at given paths.")

# try get models
try:
    encoder_model, decoder_model, training_model = get_models(enc_path, dec_path, train_path)
except Exception as e:
    st.error(f"Model loading/building failed: {e}")
    st.stop()

st.sidebar.success("Models ready ✅")

# -------------------------
# Data upload (CSV) and predict
# -------------------------
st.subheader("Upload CSV (must contain 'tag_value' column)")
csv_file = st.file_uploader("Upload CSV file (or leave to use local file)", type=["csv"])

if csv_file is None:
    st.info("Upload CSV to run prediction.")
    st.stop()

try:
    df = pd.read_csv(csv_file)
except Exception as e:
    st.error(f"Cannot read CSV: {e}")
    st.stop()

if 'tag_value' not in df.columns:
    st.error("CSV must contain 'tag_value' column.")
    st.stop()

# determine input_len and features from encoder model input shape if possible
enc_in_shape = encoder_model.input_shape  # e.g. (None, timesteps, features) or list
if isinstance(enc_in_shape, tuple):
    try:
        in_timesteps = int(enc_in_shape[1])
        in_features = int(enc_in_shape[2])
    except Exception:
        in_timesteps = 60
        in_features = 1
else:
    # sometimes input_shape is a list
    try:
        in_timesteps = int(enc_in_shape[0][1])
        in_features = int(enc_in_shape[0][2])
    except Exception:
        in_timesteps = 60
        in_features = 1

input_len = st.sidebar.number_input("Input length (detected from model)", value=in_timesteps, min_value=1)
output_len = st.sidebar.number_input("Output length (horizon)", value=60, min_value=1)

# prepare series and pad if needed
series = df['tag_value'].astype(float).values
if series.size < input_len:
    st.warning(f"Series length {series.size} < input_len {input_len} → padding with last value.")
    pad = np.full((input_len - series.size,), series[-1] if series.size>0 else 0.0)
    series = np.concatenate([pad, series])

last_seq = series[-input_len:].reshape(-1,1)
# scale
try:
    scaled = scaler.transform(last_seq)
except Exception as e:
    st.error(f"Scaler.transform failed: {e}")
    st.stop()

encoder_input = scaled.reshape(1, input_len, in_features)

# ---------- robust inference ----------
def robust_seq2seq_predict(encoder_input, encoder_model, decoder_model, output_len):
    # encode
    enc_out = encoder_model.predict(encoder_input)
    if isinstance(enc_out, (list,tuple)) and len(enc_out) >= 2:
        state_h, state_c = enc_out[0], enc_out[1]
    elif isinstance(enc_out, np.ndarray):
        # if single vector, try split
        if enc_out.shape[-1] % 2 == 0:
            half = enc_out.shape[-1]//2
            state_h = enc_out[:, :half]
            state_c = enc_out[:, half:]
        else:
            raise RuntimeError("Encoder output ambiguous; expected two states.")
    else:
        raise RuntimeError("Unexpected encoder output type.")

    # ensure batch dim
    if state_h.ndim == 1:
        state_h = state_h.reshape(1, -1)
    if state_c.ndim == 1:
        state_c = state_c.reshape(1, -1)

    # create initial decoder input: zeros or last input value
    # try to use last value from encoder input as start token
    try:
        start_val = encoder_input[0, -1, 0]
        target_seq = np.array(start_val).reshape(1,1,1)
    except Exception:
        target_seq = np.zeros((1,1,in_features))

    preds_scaled = []
    for i in range(output_len):
        # ensure shapes
        target_seq = target_seq.reshape(1, target_seq.shape[1], target_seq.shape[2])
        # call decoder: common signature [target_seq, state_h, state_c]
        try:
            yhat, h, c = decoder_model.predict([target_seq, state_h, state_c])
        except Exception as e1:
            # try as [target_seq] + list(enc_out)
            try:
                yhat, h, c = decoder_model.predict([target_seq] + list(enc_out))
            except Exception as e2:
                raise RuntimeError(f"Decoder predict failed. Tried two orders. Errors:\n1) {e1}\n2) {e2}")

        # extract predicted value (support shapes (1,1,features) or (1,t,features))
        if yhat.ndim == 3:
            val = yhat[0, -1, :]  # last timestep
        elif yhat.ndim == 2:
            val = yhat[0, :]
        else:
            val = np.array(yhat).reshape(-1)

        preds_scaled.append(np.array(val).reshape(-1))

        # prepare next input & states
        # reshape prediction to (1,1,features)
        next_in = np.array(val).reshape(1,1,-1)
        target_seq = next_in
        state_h, state_c = h, c

    preds_scaled = np.vstack(preds_scaled)  # (output_len, features)
    return preds_scaled

try:
    scaled_preds = robust_seq2seq_predict(encoder_input, encoder_model, decoder_model, int(output_len))
    # inverse scale (handle multifeature)
    if scaled_preds.ndim == 1:
        scaled_preds = scaled_preds.reshape(-1,1)
    try:
        preds = scaler.inverse_transform(scaled_preds)
    except Exception:
        # fallback assume scaler expects shape (-1,1)
        preds = np.array([scaler.inverse_transform(p.reshape(-1,1)).flatten() for p in scaled_preds]).flatten()
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# prepare output dataframe
if 'ddate' in df.columns and pd.api.types.is_datetime64_any_dtype(df['ddate']):
    last_date = pd.to_datetime(df['ddate'].iloc[-1])
    freq = pd.to_timedelta(df['ddate'].diff().median())
    if pd.isnull(freq):
        freq = pd.Timedelta(seconds=10)
    future_dates = [last_date + (i+1)*freq for i in range(int(output_len))]
    out_df = pd.DataFrame({"ddate": future_dates, "pred_tag_value": preds.flatten()})
else:
    out_df = pd.DataFrame({"index": np.arange(len(series), len(series)+int(output_len)), "pred_tag_value": preds.flatten()})

st.subheader("Predictions")
st.dataframe(out_df)

# plot
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(np.arange(len(series))[-200:], series[-200:], label="historical")
ax.plot(np.arange(len(series), len(series)+int(output_len)), preds.flatten(), 'rx-', label="predicted")
ax.legend(); ax.grid(True)
st.pyplot(fig)
