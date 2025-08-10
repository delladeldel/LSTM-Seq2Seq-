%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load the scaler
try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Scaler file not found. Please upload 'scaler.pkl'.")
    st.stop()

# Load the models
try:
    # Load using the native Keras format
    encoder_model = tf.keras.models.load_model("seq2seqencodermodelfix.keras", compile=False)
    decoder_model = tf.keras.models.load_model("seq2seqdecodermodelfix.keras", compile=False)
except Exception as e:
    st.error(f"Error loading models: {e}. Please ensure 'seq2seqencodermodelfix.keras' and 'seq2seqdecodermodelfix.keras' are in the same directory.")
    st.stop()


input_len = 60
output_len = 60
n_features = 1
latent_dim = 64 # Ensure this matches your model's latent_dim

def predict_future(input_seq, output_len=60):
    # Encode input as initial state
    states_value = encoder_model.predict(input_seq)

    # Initialize decoder input (first step)
    # The decoder expects input in the shape (batch_size, timesteps, features)
    # For the first step, timesteps is 1
    target_seq = np.zeros((input_seq.shape[0], 1, n_features))

    predictions_scaled = []

    for _ in range(output_len):
        # Predict next token
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Store prediction
        # Take the first prediction from the output_tokens (batch size 1, timestep 1)
        predictions_scaled.append(output_tokens[0, 0, 0])

        # Update target_seq for the next step
        # The next input to the decoder is the prediction we just made
        target_seq = np.zeros((input_seq.shape[0], 1, n_features))
        target_seq[0, 0, 0] = output_tokens[0, 0, 0]


        # Update state
        states_value = [h, c]

    return np.array(predictions_scaled).reshape(-1, 1)

st.title("Time Series Prediction with Seq2Seq LSTM")

st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file (should contain 'ddate' and 'tag_value' columns)", type=["csv"])

df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['ddate'] = pd.to_datetime(df['ddate'])
    df = df.sort_values('ddate')
    st.sidebar.success("File uploaded successfully!")

if df is not None:
    st.subheader("Original Data (Last 200 points)")
    st.write(df[['ddate', 'tag_value']].tail(200))

    if len(df) >= input_len:
        st.subheader(f"Predicting next {output_len} steps")

        # Get the last input sequence from the original data
        last_sequence_full = df['tag_value'].values[-input_len:].reshape(-1, 1)

        # Normalisasi data
        last_sequence_scaled = scaler.transform(last_sequence_full)

        # Reshape for model input (batch_size, timesteps, features)
        encoder_input_future = last_sequence_scaled.reshape((1, input_len, n_features))

        # Run prediction
        future_predictions_scaled = predict_future(encoder_input_future, output_len)
        future_predictions = scaler.inverse_transform(future_predictions_scaled)

        # Create timestamps for future predictions
        last_date = df['ddate'].iloc[-1]
        # Assuming 10 second intervals - adjust if your data has different frequency
        future_dates = pd.to_datetime([last_date + pd.Timedelta(seconds=10 * (i+1)) for i in range(output_len)])

        # Create DataFrames for plotting
        historical_df = df[['ddate', 'tag_value']].tail(200).copy() # Display last 200 historical points
        future_df = pd.DataFrame({'ddate': future_dates, 'tag_value': future_predictions.flatten()})

        st.subheader("Historical Data and Future Predictions")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(historical_df['ddate'], historical_df['tag_value'], label='Historical Data')
        ax.plot(future_df['ddate'], future_df['tag_value'], label='Future Predictions', color='red', marker='x')
        ax.set_title('Historical Data vs Future Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('tag_value')
        ax.legend()
        ax.grid(True)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.subheader("Future Predictions Data")
        st.write(future_df)

    else:
        st.warning(f"Not enough data to create an input sequence of length {input_len}. Please upload a CSV with at least {input_len} data points.")
