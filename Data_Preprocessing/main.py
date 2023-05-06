from meta_config import conf_dict
conf_dict["mode"] = "live"
# conf_dict["mode"] = "eval"

from Data_Preprocessing.data_preprocessing import DataPreprocessing
from Data_Preprocessing.frames_to_sample import FeatureType

if __name__ == "__main__":


    if conf_dict["mode"] == "live":
        print("LIVE MODE")
        DataPreprocessing(source_dir="training_data", feature_type=FeatureType.SYNTHETIC, data_augmentation=True)
    elif conf_dict["mode"] == "eval":
        print("EVAL MODE")
        DataPreprocessing(source_dir="training_data", feature_type=FeatureType.SYNTHETIC, data_augmentation=True, eval_mode=True)


    # DataPreprocessing(source_dir="validation_data", feature_type=FeatureType.SYNTHETIC, data_augmentation=False)
    # DataPreprocessing(source_dir="test_data", feature_type=FeatureType.SYNTHETIC, data_augmentation=False)
