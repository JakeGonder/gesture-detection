from meta_config import conf_dict

# conf_dict["mode"] = "live"
conf_dict["mode"] = "eval"

from Data_Preprocessing.frames_to_sample import FeatureType
from ML.network import Network
from Training.train import Training

if __name__ == "__main__":
    if conf_dict["mode"] == "live":
        network = Training(train_data_dir="training_data",
                           feature_type=FeatureType.SYNTHETIC,
                           show_plots=True)
    elif conf_dict["mode"] == "eval":
        network = Training(train_data_dir="eval_training_data",
                           feature_type=FeatureType.SYNTHETIC,
                           show_plots=True, eval_mode=True)

    # network = Training(train_data_dir="training_data", feature_type=FeatureType.SYNTHETIC, show_plots=True)

    # network_loaded = Network.create_with_hyper_params("Training/SYNTHETIC_features/")
    # test_network(network=network_loaded, feature_type=FeatureType.SYNTHETIC, source_dir="test_data")
