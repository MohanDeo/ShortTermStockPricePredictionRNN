from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam


def build_hyper_lstm_model(hp):
    model = Sequential()

    # Hyperparameter for number of layers
    num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)

    # Input shape
    sequence_length = hp.Fixed('sequence_length', 60)
    num_features = hp.Fixed('num_features', 1)
    input_shape = (sequence_length, num_features)

    for i in range(num_layers):
        # Hyperparameters for units and dropout
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)

        return_sequences = True if i < num_layers - 1 else False

        if i == 0:
            model.add(LSTM(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(LSTM(units=units, return_sequences=return_sequences))

        model.add(Dropout(rate=dropout_rate))

    # Output layer
    model.add(Dense(1))

    # Compile the model
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model
