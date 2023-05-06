from meta_config import conf_dict  # ensures correct config is used everywhere
conf_dict["mode"] = "eval"

import os
from collections import defaultdict

import pandas as pd
import numpy as np

from Data_Preprocessing.frames_to_sample import SampleFactory, FeatureType
from Prediction_Mode.helper import get_label_and_probability_for_sample
from ML.network import Network
from config_eval import *


def find_frame_index_ms_ago(timestamps, millis):
    """ Returns the index of the first timestamp (searching backwards from the newest, which is longer than
    the passed milliseconds ago, returns -1 if no timestamp with a large enough delta can be found"""
    i = len(timestamps) - 1
    while i >= 0:
        # search backwards from the current frame, up to a frame that was 2 seconds ago
        i = i - 1
        dif = timestamps[-1] - timestamps[i]
        if dif > millis:
            return i
    return -1


"""
TEST MODE
Generates the csv event file for any csv file in the correct format that is put into the test_data folder.
IMPORTANT!! Currently this will overwrite the results if multiple files are placed in the test_data folder!
"""


class EventGenerator:
    def __init__(self):
        pd.set_option('display.max_columns', None)
        self.coord_columns = np.array(["x", "y", "z", "confidence"])
        self.joint_names = [joint.name for joint in
                            DEFAULT_JOINT_MAPPING]  # Transforms PoseLandmark ENUM list to name list
        self.joint_ids = [FULL_JOINT_IDs[joint_name] for joint_name in DEFAULT_JOINT_MAPPING]
        self.joint_col_names = ["%s_%s" % (joint_name, coord) for joint_name in self.joint_names for coord in
                                self.coord_columns]

        self.INFERENCE_MIN_GESTURE_RATIO = 0.01
        self.joint_mapping = LIVE_INFERENCE_JOINT_MAPPING
        self.gesture_mapping = EVALUATION_GESTURE_MAPPING
        self.feature_type = FeatureType.SYNTHETIC
        self.sample_factory = SampleFactory(joint_mapping=self.joint_mapping, feature_type=self.feature_type)

        newest_file = "../Evaluation_Mode/SYNTHETIC_features/asave/eval_100percent_window13lookback05.npz"
        print(newest_file)
        self.network = Network.create_with_hyper_params(newest_file)
        self.print_info = False

        self.timestamps = []
        self.frames = []
        self.events = []
        self.start_blocking = False
        self.detections = []
        self.detection_timestamps = []
        self.command_counts = defaultdict(int)
        self.last_command_end_pos = 0

    def reset_fields(self):
        self.timestamps = []
        self.frames = []
        self.events = []
        self.start_blocking = False
        self.detections = []
        self.detection_timestamps = []
        self.command_counts = defaultdict(int)
        self.last_command_end_pos = 0

    def generate_events_for_file(self, file_path):

        try:
            self.reset_fields()
            frames = pd.read_csv(file_path, index_col="timestamp")
            frames.index = frames.index.astype(int)
            # removes rows after the max value that have timestamp zero
            frames = frames.loc[:frames.index.max()]
            frames.columns = frames.columns.str.lower()

            for index, frame in frames.iterrows():
                self.frames.append(frame)
                self.timestamps.append(index)
                self.events.append("idle")
                self.process_data()

            result = {"timestamp": self.timestamps, "events": self.events}
            df = pd.DataFrame(result)
            path = "../Test_Mode/results/performance_results.csv"
            df.to_csv(path, index=False)
            print(f"CSV results saved to {path}")
        except:
            print("Error reading CSV file, make sure it's in the correct format and there is no other files in the test_data directory.")

    def calculate_command(self):
        # Check if the current data fills the window_size and set the start position

        if self.last_command_end_pos != 0:
            if (self.timestamps[-1] - self.last_command_end_pos) \
                    < 1.5 * WINDOW_SIZE_MS:
                return
        command = self.detections[-1]
        if command != "idle":
            self.last_command_end_pos = self.timestamps[-1]
            self.start_blocking = True
            self.detections = ['idle' for _ in range(len(self.detections))]
            return command
        else:
            return None

    def process_data(self):
        # Use all frames between current frame and WINDOW_SIZE ago as input sample,
        # and continue if there are not enough frames to fill up one window
        start_pos = find_frame_index_ms_ago(self.timestamps, WINDOW_SIZE_MS)
        if start_pos == -1:
            return

        # Get the frames for one window
        frames = self.frames[start_pos:]
        frames = pd.DataFrame(frames)[USED_COLUMNS]
        frames.index.name = "timestamp"
        frames.index = pd.to_timedelta(frames.index, unit='ms')

        # Create a sample for the frames
        sample = self.sample_factory.create_sample(frames)
        detected_label, probability = get_label_and_probability_for_sample(self.network, self.gesture_mapping, sample)

        if self.start_blocking and detected_label == "idle":
            self.start_blocking = False

        debug = ""
        if probability < 0.8 and detected_label != "idle":
            debug = " previously " + detected_label
            detected_label = "idle"

        if self.start_blocking:
            detected_label = "idle"

        if self.print_info:
            print(f"{detected_label} + {probability}{debug}")

        self.detections.append(detected_label)
        self.detection_timestamps.append(self.timestamps[-1])

        command = self.calculate_command()
        if command is not None:
            if self.print_info:
                print("-------------------------" + command + "------------------------------")
            self.command_counts[command] += 1
            event_index = find_frame_index_ms_ago(self.timestamps, int(WINDOW_SIZE_MS * 0.5))
            if event_index == -1:
                print("this should not happen")
                self.events[-20] = command
            else:
                self.events[event_index] = command


if __name__ == "__main__":
    e = EventGenerator()
    e.print_info = True

    e.generate_events_for_file("test_data/" + os.listdir("test_data")[0])
