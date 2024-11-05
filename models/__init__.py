from .rnn_model import build_rnn_model
from .lstm_model import build_lstm_model

def get_model(model_name, input_shape, params):
    if model_name == 'rnn':
        return build_rnn_model(input_shape, params)
    elif model_name == 'lstm':
        return build_lstm_model(input_shape, params)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
