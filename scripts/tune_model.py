import argparse
import os
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from models import get_hypermodel
from data.data_preprocessing import load_data, preprocess_data, create_sequences
from utils.utils import load_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for the model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--model', type=str, choices=['lstm', 'rnn'], required=True, help='Type of model to tune.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seed
    set_seed(config.get('seed', 42))

    # Load hypermodel based on model type
    build_hypermodel = get_hypermodel(args.model)

    # Load and preprocess data
    data_path = config['data_path']
    raw_data = load_data(data_path)
    processed_data = preprocess_data(raw_data, config['data_params'])
    x_train, y_train, x_val, y_val, num_features = create_sequences(processed_data, config['data_params'])

    # Define tuner
    tuner = RandomSearch(
        build_hypermodel,
        objective='val_loss',
        max_trials=20,
        executions_per_trial=1,
        directory=f'hyperparameter_tuning_{args.model}',
        project_name=f'{args.model}_tuning'
    )

    # Run tuning
    tuner.search(
        x=x_train,
        y=y_train,
        epochs=config.get('tune_epochs', 50),
        validation_data=(x_val, y_val),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        ]
    )

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the best model and train it
    model = tuner.hypermodel.build(best_hps)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=config.get('final_epochs', 100),
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),
            keras.callbacks.ModelCheckpoint(f'results/models/best_{args.model}_model.h5', save_best_only=True)
        ]
    )

    # Save the final model
    model.save(f'results/models/final_{args.model}_model.h5')
    print(f"Tuning and training completed. Best model saved to results/models/final_{args.model}_model.h5")


if __name__ == '__main__':
    main()