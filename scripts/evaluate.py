import argparse
import os
import tensorflow as tf
from data.data_preprocessing import load_data, preprocess_data, create_sequences
from utils.utils import load_config
from utils.visualization import plot_predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate the trained model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    parser.add_argument('--model', type=str, choices=['lstm', 'rnn'], required=True, help='Type of model to evaluate.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved model file.')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Load and preprocess data
    data_path = config['data_path']
    raw_data = load_data(data_path)
    processed_data = preprocess_data(raw_data, config['data_params'])
    x_train, y_train, x_val, y_val, x_test, y_test, num_features = create_sequences(processed_data,
                                                                                    config['data_params'], test=True)

    # Load model
    model = tf.keras.models.load_model(args.model_path)

    # Evaluate model
    test_loss, test_mae = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

    # Generate predictions
    y_pred = model.predict(x_test)

    # Plot predictions
    os.makedirs('results/figures/', exist_ok=True)
    plot_path = f"results/figures/{args.model}_predictions.png"
    plot_predictions(y_test, y_pred, plot_path)
    print(f"Prediction plot saved to {plot_path}")


if __name__ == '__main__':
    main()