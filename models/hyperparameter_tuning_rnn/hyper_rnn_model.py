import tensorflow as tf
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.models import Sequential


def build_hyper_rnn_model(hp):
    model = Sequential()

    num_layers = hp.Int('num_layers', min_value=1, max_value=5, step=1)
    input_shape = (hp.Fixed('sequence_length', 60), hp.Fixed('num_features', 1))

    for i in range(num_layers):
        units = hp.Int(f'units_{i}', min_value=32, max_value=256, step=32)
        dropout_rate = hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.5, step=0.1)

        return_sequences = True if i < num_layers - 1 else False

        if i == 0:
            model.add(SimpleRNN(units=units, return_sequences=return_sequences, input_shape=input_shape))
        else:
            model.add(SimpleRNN(units=units, return_sequences=return_sequences))

        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(1))

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mae'])

    return model
