from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess_data(df, sequence_length=60):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)

    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i, 0])  # Predict 'Close' price
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler
