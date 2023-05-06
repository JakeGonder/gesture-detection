import os
from typing import List

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from Data_Acquisition.csv_data_writer import CSVDataWriter
from config_live import DATASET_CREATION_GESTURES, DEFAULT_JOINT_MAPPING

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose


def find_files_with_prefix_recursive(dir_path: str, prefix: str) -> List[str]:
    """
    Recursively searches for files with a given prefix in a directory and its subdirectories, and returns a list of file
    names without their extension that match the prefix and have the .txt extension.

    :param dir_path: The path to the directory to search in.
    :param prefix: The prefix to search for at the beginning of the file name.
    :return: A list of file names that match the specified prefix and have the .txt extension, without their extension.
    :raises TypeError: If either `dir_path` or `prefix` is not a string.
    :raises FileNotFoundError: If `dir_path` does not exist or is not a directory.
    """
    files = []
    for _, _, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.startswith(prefix) and filename.endswith('.txt'):
                files.append(os.path.splitext(filename)[0])
    return files


def read_csv_as_df(csv_path, timestamp_idx=True):
    dataFrame = pd.read_csv(csv_path)
    if timestamp_idx:
        dataFrame["timestamp"] = pd.to_timedelta(dataFrame["timestamp"], unit="ms")
        dataFrame = dataFrame.set_index("timestamp")
        dataFrame.index = dataFrame.index.rename("timestamp")
    return dataFrame


class DataAcquisition:
    def __init__(self, source_dir, gesture_mapping=DATASET_CREATION_GESTURES, joint_mapping=DEFAULT_JOINT_MAPPING):
        # Directories
        self.data_source_dir = f"{source_dir}/labels/"
        self.unlabelled_csv_dir = f"{source_dir}/video_data/"
        self.labelled_csv_dir = f"{source_dir}/video_data_labelled/"
        self.gesture_mapping = gesture_mapping
        self.joint_mapping = joint_mapping

    def init_dataset_structure(self):
        # Check if data_source_dir exists, create it if it doesn't exist
        if not os.path.exists(self.data_source_dir):
            os.makedirs(self.data_source_dir)

        # Check and create unlabeled_csv_dir
        if not os.path.exists(self.unlabelled_csv_dir):
            os.makedirs(self.unlabelled_csv_dir)

        # Check and create labeled_csv_dir
        if not os.path.exists(self.labelled_csv_dir):
            os.makedirs(self.labelled_csv_dir)

        # Create subdirectory for each gesture from gesture_mapping
        for gesture in self.gesture_mapping:
            gesture_dir = os.path.join(self.data_source_dir, gesture)
            if not os.path.exists(gesture_dir):
                os.makedirs(gesture_dir)

    def load_the_data(self, gesture: [str] = None):
        self.validate_directories()

        gestures_to_use = self.gesture_mapping if gesture is None else gesture

        for gesture in gestures_to_use:
            gesture_vid_paths = self.find_mp4_files_with_prefix_recursive(prefix=gesture)

            for file in gesture_vid_paths:
                self.process_vid(file=file, gesture=gesture)

    def validate_directories(self):
        if not os.path.exists(self.data_source_dir):
            raise NotADirectoryError("The data source directory was not found.")
        if not os.path.exists(self.unlabelled_csv_dir):
            raise NotADirectoryError("The unlabelled csv directory was not found.")
        if not os.path.exists(self.labelled_csv_dir):
            raise NotADirectoryError("The labelled csv directory was not found.")

    def find_mp4_files_with_prefix_recursive(self, prefix):
        mp4_files = []
        for _, _, filenames in os.walk(self.data_source_dir):
            for filename in filenames:
                if filename.startswith(prefix) and filename.endswith('.mp4'):
                    mp4_files.append(filename)
        return mp4_files

    # TODO: Optimize with multi core processing @Micha
    def process_vid(self, file, gesture):
        csv_writer = CSVDataWriter(joint_mapping=self.joint_mapping)
        result_csv_name = f"{self.unlabelled_csv_dir}/{os.path.splitext(file)[0]}.csv"
        vid_source = cv2.VideoCapture(f"{self.data_source_dir}/{gesture}/{file}")

        frame_counter = 0
        success = True

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
            progress_bar = tqdm(total=vid_source.get(cv2.CAP_PROP_FRAME_COUNT), desc=f'Processing {file}', unit='frames')
            while vid_source.isOpened() and success:

                success, image = vid_source.read()

                if not success:
                    break

                frame_counter += 1
                progress_bar.update(1)

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                pose_landmarks = pose.process(image).pose_landmarks

                # Read data
                csv_writer.read_data(data=pose_landmarks, timestamp=vid_source.get(cv2.CAP_PROP_POS_MSEC))

        csv_writer.to_csv(result_csv_name)
        vid_source.release()

    def label_the_data(self, debug_print=False):
        # Variables for the statistical report
        total_idle_percentage = []
        total_count = 0
        total_idle_count = 0
        label_cnt_per_gesture = {}
        avg_label_len_per_gesture = {}
        avg_sample_lengths = {}
        avg_label_len_overall = np.array([])
        avg_label_len_gestures_seconds = {}

        for gesture in tqdm(self.gesture_mapping):
            files = find_files_with_prefix_recursive(f"{self.data_source_dir}{gesture}/", gesture)

            if len(files) == 0:
                continue

            label_cnt_per_gesture[str(gesture).title()] = 0
            avg_label_len_per_gesture[str(gesture).title()] = []
            avg_label_len_gestures_seconds[str(gesture).title()] = []

            for file in files:
                # ===== Load Data =====
                frames = read_csv_as_df(csv_path=f"{self.unlabelled_csv_dir}{file}.csv")
                frames["ground_truth"] = "idle"  # Default lable is 'idle'

                labels = pd.read_csv(f"{self.data_source_dir}{gesture}/{file}.txt",
                                     sep="\t",
                                     header=None,
                                     usecols=[3, 5, 8],
                                     names=["start", "end", "label"])

                # Append amount of labels for this gesture video
                label_cnt_per_gesture[str(gesture).title()] += len(labels)

                # Convert start and end timestamps into a datetime-datatype
                labels["start"] = pd.to_timedelta(labels["start"], unit="s")
                labels["end"] = pd.to_timedelta(labels["end"], unit="s")
                for idx, label in labels.iterrows():
                    annotated_frames = (frames.index >= label["start"]) & (frames.index <= label["end"])
                    annotation_len = label["end"] - label["start"]
                    avg_label_len_overall = np.append(avg_label_len_overall, annotation_len)
                    avg_label_len_gestures_seconds[str(gesture).title()].append(annotation_len.total_seconds())
                    frames.loc[annotated_frames, "ground_truth"] = label["label"]

                # ===== Save result =====
                # Convert timestamp index to integer index
                frames.index = frames.index.astype(int) // 1_000_000
                frames.to_csv(f"{self.labelled_csv_dir}{file}.csv", index=True)

                # ===== Report =====
                total_gesture_count = frames.shape[0]
                total_count += total_gesture_count

                non_idle_count = frames[~frames['ground_truth'].str.contains('idle')]['ground_truth'].count()
                non_idle_percentage = round(non_idle_count / total_gesture_count, 2)

                # Append average label length for this gesture video
                avg_label_len_per_gesture[str(gesture).title()].append(non_idle_count / len(labels))

                idle_count = frames[frames['ground_truth'].str.contains('idle')]['ground_truth'].count()
                total_idle_count += idle_count
                idle_percentage = round(idle_count / total_gesture_count, 2)
                total_idle_percentage.append(idle_percentage)

                if debug_print:
                    print(f"\nTotal Entries: {total_gesture_count}")
                    print(f"{str(file).title()}: {non_idle_percentage}")
                    print(f"Idle: {idle_percentage}")

            avg_sample_lengths[str(gesture).title()] = np.array(
                avg_label_len_per_gesture[str(gesture).title()]).sum() / len(
                avg_label_len_per_gesture[str(gesture).title()])

        if debug_print:
            print("========================")
        print(f"\n===>>> Overall Evaluation <<<===")
        print(f"Total Entries: {total_count}")
        print(f"Total Gesture Entries: {total_count - total_idle_count}")
        print(f"Total Idle Entries: {total_idle_count}")
        print(f"Average Gesture Entries: {round((1 - (np.array(total_idle_percentage).mean())) * 100, 2)} %")
        print(f"Average Idle Entries: {round(np.array(total_idle_percentage).mean() * 100, 2)} %")
        print(f"Overall Annotation Count: {len(avg_label_len_overall)}")
        print(f"Overall Average Annotation Time: {'{:.3f}'.format(avg_label_len_overall.mean().total_seconds())}s")
        print(f"Overall Median Annotation Time: {'{:.3f}'.format(np.median(avg_label_len_overall).total_seconds())}s")
        print(f"Overall Minimum Annotation Time: {'{:.3f}'.format(np.amin(avg_label_len_overall).total_seconds())}s")
        print(f"Overall Maximum Annotation Time: {'{:.3f}'.format(np.amax(avg_label_len_overall).total_seconds())}s")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 8))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
                  'tab:olive', 'tab:cyan', '#F0E442']
        ax1.pie(label_cnt_per_gesture.values(), labels=label_cnt_per_gesture.keys(), autopct='%1.1f%%', colors=colors)
        ax2.pie(avg_sample_lengths.values(), labels=avg_sample_lengths.keys(), autopct='%1.1f%%', colors=colors)
        ax1.set_title('Gesture Amounts')
        ax2.set_title('Average Gesture Annotation Lengths')
        plt.show()

        plot_gestures = {"Swipe_Right": "SR",
                            "Swipe_Left": "SL",
                            "Swipe_Up": "SU",
                            "Swipe_Down": "SD",
                            "Rotate_Right": "RR",
                            "Rotate_Left": "RL",
                            "Spread": "Spread",
                            "Pinch": "Pinch",
                            "Flip_Table": "Flip",
                            "Spin": "Spin",
                            "Point": "Point"}

        x_values = []
        y_values = []
        for key, values in avg_label_len_gestures_seconds.items():
            for value in values:
                x_values.append(plot_gestures[key])
                y_values.append(value)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(x_values, y_values)
        ax1.set_xticklabels(list(avg_label_len_gestures_seconds.keys()), rotation=90)
        ax1.set_title('Gesture Length Distribution (Scatterplot)')
        ax1.set_xlabel('Gesture Names')
        ax1.set_ylabel('Gesture Lengths (seconds)')
        plt.tight_layout()

        plt.savefig('gesture_length_scatterplot.png')

        fig, ax2 = plt.subplots(figsize=(8, 5))
        boxplot_values = [avg_label_len_gestures_seconds[k] for k in avg_label_len_gestures_seconds]
        boxplot_labels = [k for k in avg_label_len_gestures_seconds]
        ax2.boxplot(boxplot_values)
        ax2.set_xticklabels(boxplot_labels, rotation=45, ha='right', va='top')
        ax2.set_title('Gesture Length Distribution (Boxplot)')
        ax2.set_xlabel('Gesture Names')
        ax2.set_ylabel('Gesture Lengths (seconds)')
        ax2.tick_params(axis='x', pad=10)
        plt.tight_layout()

        plt.savefig('gesture_length_boxplot.svg', format='svg')

        plt.show()

        avg_label_len_overall = [x.total_seconds() for x in avg_label_len_overall]

        avg_len = np.mean(avg_label_len_overall)
        med_len = np.median(avg_label_len_overall)

        max_len = max(avg_label_len_overall)
        bin_edges = np.arange(0.4, max_len + 0.1, 0.1)

        plt.hist(avg_label_len_overall, bins=bin_edges, edgecolor='white')

        plt.axvline(x=avg_len, color='r', linestyle='-', label=f'Mean: {avg_len:.2f}')  # Add mean line
        plt.axvline(x=med_len, color='g', linestyle='-', label=f'Median: {med_len:.2f}')  # Add median line
        plt.legend()

        plt.title('Distribution of Annotations (Histogram)')
        plt.xlabel('Annotation Length (s)')
        plt.ylabel('Annotation Frequency')
        plt.savefig('distribution_gesture_length.png')
        plt.show()