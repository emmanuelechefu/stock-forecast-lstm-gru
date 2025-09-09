from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),                     # Add Input layer
        LSTM(50, return_sequences=False),             # No more input_shape here
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),                     # Add Input layer
        GRU(50, return_sequences=False),              # No more input_shape here
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
