import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml


def plot_predictions(y_true, y_pred, save_path):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title('Actual vs Predicted Prices')
    plt.savefig(save_path)
    plt.close()


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
