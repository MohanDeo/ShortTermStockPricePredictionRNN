from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, params):
    model = Sequential()
    for i in range(params.get('num_layers', 1)):
        return_sequences = i < params.get('num_layers', 1) - 1
        model.add(LSTM(
            units=params.get('units_per_layer', 64),
            return_sequences=return_sequences,
            input_shape=input_shape if i == 0 else None))
        model.add(Dropout(params.get('dropout', 0.2)))
    model.add(Dense(1))
    return model