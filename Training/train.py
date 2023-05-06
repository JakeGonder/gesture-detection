from os.path import exists

import numpy as np

from ML.network import Network
from meta_config import conf_dict
if conf_dict["mode"] == "live":
    from config_live import *
else:
    from config_eval import *


def load_data(source_dir, feature_type):
    """Loads data from a given 'Data Preprocessing' source directory."""
    X_path = f"Data_Preprocessing/{source_dir}/{feature_type.name}_features/X.npy"
    Y_path = f"Data_Preprocessing/{source_dir}/{feature_type.name}_features/Y.npy"
    if not exists(X_path) or not exists(Y_path):
        raise FileNotFoundError(
            f"For the given directory Data_Preprocessing/{source_dir}/{feature_type.name}_features/ either X.npy "
            f"or Y.npy does not exist.")
    X = np.load(X_path, allow_pickle=True)
    Y = np.load(Y_path, allow_pickle=True)
    return X, Y


def test_network(network, feature_type, source_dir=""):
    # Load test dataset
    X_test, y_test = load_data(source_dir=source_dir, feature_type=feature_type)

    # Predict
    y_test_pred = network.predict(X=X_test, apply_feature_scaling=True)

    # Evaluate
    network.evaluate(y_pred=y_test_pred, y=y_test)


class Training:

    def __init__(self, feature_type, train_data_dir, val_data_dir="", hyper_param_file="", show_plots=True,
                 eval_mode=False):
        self.train_network(feature_type, train_data_dir, val_data_dir, hyper_param_file, show_plots, eval_mode)

    def train_network(self, feature_type, train_data_dir, val_data_dir="", hyper_param_file="", show_plots=True,
                      eval_mode=False):
        # Directory to save scaling parameter and thetas
        target_dir = f"Evaluation_Mode/{feature_type.name}_features" if eval_mode else f"Training/{feature_type.name}_features"

        gesture_mapping = EVALUATION_GESTURE_MAPPING if eval_mode else DEFAULT_GESTURE_MAPPING

        # Load data
        X, Y = load_data(source_dir=train_data_dir, feature_type=feature_type)

        # Shuffle data
        np.random.seed(NETWORK_SEED)
        perm = np.random.permutation(len(X))
        X = X[perm]
        Y = Y[perm]

        # Validation Split
        if val_data_dir == "":
            # If no validation data directory is given, split the training data
            split_index = int(len(X) * TRAINING_VALIDATION_SPLIT)
            X_train = X[split_index:]
            Y_train = Y[split_index:]
            X_val = X[:split_index]
            Y_val = Y[:split_index]
        else:
            X_train, Y_train = X, Y
            X_val, Y_val = load_data(source_dir=val_data_dir, feature_type=feature_type)

        if hyper_param_file:
            # Load existing network
            hyper_param_path = f"{target_dir}/{hyper_param_file}"
            network = Network.create_with_hyper_params(source_dir=hyper_param_path)
        else:
            layer_shape = [X.shape[1]]
            for layer in NETWORK_HIDDEN_LAYERS_SHAPE:
                layer_shape.append(layer)
            layer_shape.append(len(gesture_mapping))

            # Create a new network
            network = Network(layer_shape=layer_shape, seed=NETWORK_SEED)

        # Construct a list of feature names
        feature_names = []
        for frame in range(1, FRAMES_PER_SAMPLE + 1):
            for feature in SYNTHETIC_FEATURE_NAMES:
                feature_names.append(f"{frame}_{feature}")

        # Train the network
        network.train(feature_names=feature_names,
                      hidden_layer_shape=NETWORK_HIDDEN_LAYERS_SHAPE,
                      iterations=TRAINING_ITERATIONS,
                      alpha=TRAINING_ALPHA,
                      X_train=X_train,
                      y_train=Y_train,
                      X_val=X_val,
                      y_val=Y_val,
                      show_plots=show_plots,
                      log_intervals=TRAINING_LOG_INTERVALS,
                      use_feature_scaling=TRAINING_USE_FEATURE_SCALING,
                      lambda_regularization=TRAINING_LAMBDA,
                      use_regularization=TRAINING_USE_REGULARIZATION,
                      class_labels=gesture_mapping,
                      adjust_alpha=ADJUST_ALPHA,
                      pca_threshold=PCA_THRESHOLD)

        network.save(target_dir=target_dir)

        return network
