from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout

def build_rnn_model(input_shape, params):
    model = Sequential()

    num_layers = params.get('num_layers', 1)
    units_per_layer = params.get('units_per_layer', 64)
    dropout_rate = params.get('dropout', 0.2)
    activation = params.get('activation', 'tanh')
    recurrent_dropout = params.get('recurrent_dropout', 0.0)
    bidirectional = params.get('bidirectional', False)

    for i in range(num_layers):
        return_sequences = i < num_layers - 1
        layer = SimpleRNN(
            units=units_per_layer,
            activation=activation,
            return_sequences=return_sequences,
            recurrent_dropout=recurrent_dropout,
            input_shape=input_shape if i == 0 else None)

        if bidirectional:
            from tensorflow.keras.layers import Bidirectional
            layer = Bidirectional(layer)

        model.add(layer)
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))
    return model
