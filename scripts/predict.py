import argparse
import os
import pandas as pd
import tensorflow as tf
from data.data_preprocessing import load_data, preprocess_data, create_sequences_for_prediction
from utils.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using the trained model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--model', type=str, choices=['lstm', 'rnn'], required=True, help='Type of model to use.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the new data CSV file.')
    parser.add_argument('--output_path', type=str, default='results/predictions/',
                        help='Directory to save predictions.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load and preprocess new data
    raw_data = load_data(args.data_path)
    processed_data = preprocess_data(raw_data, config['data_params'], is_training=False)
    x_pred, num_features = create_sequences_for_prediction(processed_data, config['data_params'])

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    # Make predictions
    predictions = model.predict(x_pred)

    # Save predictions
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, f'{args.model}_predictions.csv')
    pd.DataFrame(predictions, columns=['Predicted']).to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == '__main__':
    main()