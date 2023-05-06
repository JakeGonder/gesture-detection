import multiprocessing
import os
import pathlib
import time
import traceback
from collections import defaultdict

import cv2
import mediapipe as mp
from matplotlib import pyplot as plt

from Data_Acquisition.csv_data_writer import CSVDataWriter
from Data_Preprocessing.frames_to_sample import SampleFactory, FeatureType
from Prediction_Mode.helper import invalid_landmarks, find_frame_index_ms_ago, get_label_and_probability_for_sample
from Prediction_Mode.live_video_feed import WINDOW_SIZE_MS
from ML.network import Network
from config_static import LIVE_INFERENCE_JOINT_MAPPING
from config_live import INFERENCE_MIN_GESTURE_CONFIDENCE, INFERENCE_BLOCK_WINDOW_MULTIPLIER, INFERENCE_MIN_GESTURE_RATIO

script_dir = pathlib.Path(__file__).parent

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

#todo read ground truth from annotation txt
EVAL_TEST_VIDEOS = {
    "../Data_Acquisition/training_data/labels/rotate_right/rotate_right_marcel.mp4": {
        "total": 24},  # 1
    "../Data_Acquisition/training_data/labels/rotate_right/rotate_right_micha.mp4": {
        "total": 27},  # 2
    "../Data_Acquisition/training_data/labels/rotate_right/rotate_right_jan.mp4": {
        "total": 28},  # 3

    "../Data_Acquisition/training_data/labels/swipe_left/swipe_left_marcel.mp4": {
        "total": 22},  # 4
    "../Data_Acquisition/training_data/labels/swipe_left/swipe_left_micha.mp4": {
        "total": 24},  # 5
    "../Data_Acquisition/training_data/labels/swipe_left/swipe_left_jan.mp4": {
        "total": 22},  # 6

    "../Data_Acquisition/training_data/labels/swipe_right/swipe_right_marcel.mp4": {
        "total": 29},  # 7
    "../Data_Acquisition/training_data/labels/swipe_right/swipe_right_micha.mp4": {
        "total": 27},  # 8
    "../Data_Acquisition/training_data/labels/swipe_right/swipe_right_jan.mp4": {
        "total": 25},  # 9

    "../Data_Acquisition/validation_data/labels/rotate_right/rotate_right.mp4": {
        "total": 6},  # 10
    "../Data_Acquisition/validation_data/labels/swipe_right/swipe_right.mp4": {
        "total": 5},  # 11
    "../Data_Acquisition/validation_data/labels/swipe_left/swipe_left.mp4": {
        "total": 8},  # 12

    "../Data_Acquisition/test_data/labels/rotate_right/rotate_right_emma.mp4": {
        "total": 7},  # 13
    "../Data_Acquisition/test_data/labels/swipe_right/swipe_right_emma.mp4": {
        "total": 7},  # 14
    "../Data_Acquisition/test_data/labels/swipe_left/swipe_left_emma.mp4": {
        "total": 7}  # 15
}


class LiveLoopInferenceTest:

    def __init__(self):
        network_path = "../Training/SYNTHETIC_features/hyper_params_2023-04-02_02-29-52.npz"
        self.network = Network.create_with_hyper_params(network_path)

        self.joint_mapping = LIVE_INFERENCE_JOINT_MAPPING
        self.gesture_mapping = self.network.get_label_names()
        self.feature_type = FeatureType.SYNTHETIC

        self.start()

    def start(self):
        time_start = time.time()

        cpu_count = multiprocessing.cpu_count()
        results = multiprocessing.Manager().dict()

        videos_to_use = list(EVAL_TEST_VIDEOS.keys())
        videos_left = videos_to_use.copy()

        # video_path = videos_left.pop(13)
        # self.process_file(results=results, video_path=video_path)
        # exit()

        while len(results) < len(videos_to_use):
            print(f"Amount of files to use: {len(videos_to_use)}")
            num_threads = min(len(videos_left), cpu_count)
            pool = multiprocessing.Pool(num_threads)
            print(f"Creating pool of size {num_threads}")
            for _ in range(num_threads):
                video_path = videos_left.pop(0)
                print("Creating worker for " + video_path)
                pool.apply_async(self.process_file, args=(results, video_path))
            pool.close()
            pool.join()
            len_res = len(results)
            if len_res == 0:
                raise AssertionError("Results empty. Debug data preprocessing for a single file to get detailed information.")
            else:
                print(len(results))
        print("Processing done")

        for key, value in results.items():
            print(f"{os.path.basename(key)}: detected {value} out of {EVAL_TEST_VIDEOS[key]['total']}")

        time_end = time.time()
        print("Time taken for data preprocessing: ", round(time_end - time_start, 2), "s")

    def process_file(self, results, video_path):
        # Processing variables
        frame_counter = 0
        detections = []
        detection_timestamps = []
        last_command_end_pos = [-5000]

        csv_writer = CSVDataWriter(joint_mapping=self.joint_mapping, mask=[True, True, True, True])
        sample_factory = SampleFactory(joint_mapping=self.joint_mapping, feature_type=self.feature_type, eval_mode=False)

        vid_source = cv2.VideoCapture(video_path)
        command_counts = defaultdict(int)

        success = True
        try:
            with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                while vid_source.isOpened() and success:

                    success, image = vid_source.read()

                    if not success:
                        break

                    frame_counter += 1

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    pose_landmarks = pose.process(image).pose_landmarks

                    if invalid_landmarks(pose_landmarks=pose_landmarks):
                        continue

                    # Read data
                    csv_writer.read_data(data=pose_landmarks, timestamp=vid_source.get(cv2.CAP_PROP_POS_MSEC))

                    # Process data
                    self.process_data(sample_factory, csv_writer, detections, detection_timestamps, command_counts,
                                      last_command_end_pos)

            vid_source.release()
            print(f"===== Finished Video {os.path.basename(video_path)}=====")
            # results[video_path] = {
            #     os.path.basename(video_path): command_counts
            # }
            results[video_path] = command_counts
        except Exception as e:
            print(f"Exception in process_file for video {os.path.basename(video_path)}: {e}")
            traceback.print_exc()

    def process_data(self, sample_factory, csv_writer, detections, detection_timestamps, command_counts,
                     last_cmd_end_pos):
        start_pos = find_frame_index_ms_ago(csv_writer.timestamps, WINDOW_SIZE_MS)
        if start_pos == -1:
            return

        frames = csv_writer.get_frames(start_pos)
        sample = sample_factory.create_sample(frames)

        # Predict label
        detected_label, probability = get_label_and_probability_for_sample(self.network, self.gesture_mapping, sample)

        detected_label = "idle" if probability < INFERENCE_MIN_GESTURE_CONFIDENCE else detected_label

        detections.append(detected_label)
        detection_timestamps.append(csv_writer.timestamps[-1])

        command = self.get_command(self.gesture_mapping, csv_writer, detections, detection_timestamps, last_cmd_end_pos)

        if command is not None:
            # print("-------------------------" + command + "------------------------------")
            command_counts[command] += 1

    @staticmethod
    def get_command(gesture_mapping, csv_writer, detections, detection_timestamps, last_command_end_pos):
        # Check if the current data fills the window_size and set the start position
        start_pos = find_frame_index_ms_ago(detection_timestamps, WINDOW_SIZE_MS)
        if start_pos == -1:
            return

        if (csv_writer.timestamps[-1] - last_command_end_pos[0]) \
                < INFERENCE_BLOCK_WINDOW_MULTIPLIER * WINDOW_SIZE_MS:
            return

        # Get detections in given time window
        detections = detections[start_pos:]

        non_idle_detections = [detection for detection in detections if detection != "idle"]

        if len(non_idle_detections) / len(detections) >= INFERENCE_MIN_GESTURE_RATIO:
            counts = []
            for label in gesture_mapping:
                counts.append(non_idle_detections.count(label))

            most_frequent = gesture_mapping[counts.index(max(counts))]
            last_command_end_pos[0] = csv_writer.timestamps[-1]
            return most_frequent
        else:
            return None

    def report(self, command_counts):
        plt.bar(command_counts.keys(), command_counts.values())


if __name__ == "__main__":
    LiveLoopInferenceTest()
