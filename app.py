import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import pickle
import os

# Try to load existing model, create new one if not found
try:
    with open('arima_model.sav', 'rb') as file:
        model = pickle.load(file)
    st.success("Loaded existing ARIMA model")
except FileNotFoundError:
    st.warning("No saved model found - will create a new ARIMA model for predictions")
    
# Load the model from the file
with open('arima_model.sav', 'rb') as file:
    model = pickle.load(file)

st.header('Bitcoin Price Prediction Model')
st.subheader('Bitcoin Price Data')
data = pd.read_csv('dataset.csv')
data.rename(columns={"Unnamed: 0":"SNo"},inplace=True)
data.index=pd.to_datetime(data.date)
data.drop(["SNo","timestamp", "date"],axis=1,inplace=True)
df_bitcoin=data[data['crypto_name']=='Bitcoin'].copy()
df_bitcoin = df_bitcoin.reset_index().rename(columns={'index': 'date'})  # Rename during reset
st.write(df_bitcoin)

st.subheader('Bitcoin Line Chart')
# Use 'date' instead of 'index' since we renamed it
df_bitcoin_chart = df_bitcoin[['date', 'close']].copy()
st.line_chart(df_bitcoin_chart.set_index('date'))

# Prepare data for prediction
train_data = df_bitcoin[:-100]['close'].copy()
test_data = df_bitcoin[-100:]['close'].copy()

history = list(train_data)
predictions = []

# Modify the prediction loop to not rely on loaded model
for t in range(len(test_data)):
    try:
        model = SARIMAX(history, order=(1, 0, 0))
        model_fit = model.fit(disp=0)
        fc = model_fit.forecast(steps=1)[0]
        predictions.append(fc)
        history.append(test_data.iloc[t])
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        break

# Save the final model
try:
    with open('arima_model.sav', 'wb') as file:
        pickle.dump(model, file)
    st.success("Saved new ARIMA model")
except Exception as e:
    st.warning(f"Could not save model: {str(e)}")

# Create DataFrame for results with date index
predictions_df = pd.DataFrame({
    'Date': df_bitcoin['date'].iloc[-len(test_data):],
    'Original': test_data.values,
    'Predicted': predictions
})
predictions_df.set_index('Date', inplace=True)

# Display the results
st.subheader('Predicted vs Original Prices Chart')
st.line_chart(predictions_df)

