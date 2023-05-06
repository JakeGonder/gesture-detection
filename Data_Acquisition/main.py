from Data_Acquisition.data_acquisition import DataAcquisition
from config_live import FULL_JOINT_IDs

if __name__ == "__main__":
    # Acquire train/validation data with full joints
    FULL_JOINT_MAPPING = [joint for joint in list(FULL_JOINT_IDs)]

    train_data_acq = DataAcquisition(source_dir="training_data", joint_mapping=FULL_JOINT_MAPPING)
    train_data_acq.init_dataset_structure()
    # train_data_acq.load_the_data(gesture=['spin'])
    train_data_acq.label_the_data()

    val_data_acq = DataAcquisition(source_dir="validation_data", joint_mapping=FULL_JOINT_MAPPING)
    val_data_acq.init_dataset_structure()
    val_data_acq.load_the_data()
    val_data_acq.label_the_data()

    test_data_acq = DataAcquisition(source_dir="test_data", joint_mapping=FULL_JOINT_MAPPING)
    test_data_acq.init_dataset_structure()
    test_data_acq.load_the_data()
    test_data_acq.label_the_data()
