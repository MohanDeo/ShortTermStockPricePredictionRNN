import argparse
import os
import tensorflow as tf
from models import get_model
from data.data_preprocessing import load_data, preprocess_data, create_sequences
from utils.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for stock price prediction.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--model', type=str, choices=['lstm', 'rnn'], required=True, help='Type of model to train.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get('seed', 42))

    # Load and preprocess data
    data_path = config['data_path']
    raw_data = load_data(data_path)
    processed_data = preprocess_data(raw_data, config['data_params'])
    x_train, y_train, x_val, y_val, num_features = create_sequences(processed_data, config['data_params'])

    # Get model
    input_shape = (config['data_params']['sequence_length'], num_features)
    model = get_model(args.model, input_shape, config['model_params'])

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
        loss='mean_squared_error',
        metrics=['mae']
    )

    # Define callbacks
    os.makedirs('results/models/', exist_ok=True)
    model_checkpoint_path = f"results/models/best_{args.model}_model.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True)
    ]

    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=callbacks
    )

    # Save final model
    final_model_path = f"results/models/final_{args.model}_model.h5"
    model.save(final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")


if __name__ == '__main__':
    main()