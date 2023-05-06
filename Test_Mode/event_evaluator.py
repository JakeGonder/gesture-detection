from meta_config import conf_dict
conf_dict["mode"] = "eval"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Test_Mode.provided import calculator
from Test_Mode.event_generator import EventGenerator


provided_path = "../Data_Acquisition/validation_data/"
test_path = "../Data_Acquisition/test_data/"
show_plots = True

categories = ['idle', 'rotate', 'swipe_left', 'swipe_right']

gestures_in_file_names = [
    "swipe_right",
    "swipe_left",
    "rotate_right"
]
data_sets = [
    {"description": "Provided", "path": provided_path},
    {"description": "Test (Emma)", "path": test_path}
]

event_generator = EventGenerator()


def valid_filename(file_name):
    for g in gestures_in_file_names:
        if g in file_name:
            return True
    return False


def visualize_files(event_path, ground_truth_path):
    events = pd.read_csv(event_path)
    ground_truth = pd.read_csv(ground_truth_path)
    # compatibility so our regular test data works with this too
    ground_truth["ground_truth"] = ground_truth["ground_truth"].replace("rotate_right", "rotate")
    prediction_oh = np.eye(len(categories))[pd.Categorical(events.events, categories=categories).codes]
    ground_truth_oh = np.eye(len(categories))[pd.Categorical(ground_truth.ground_truth, categories=categories).codes]
    px = 1 / plt.rcParams['figure.dpi']
    fig, ax = plt.subplots(figsize=(10, (prediction_oh.shape[0] + 30) * px))
    ground_truth_oh[prediction_oh > 0] = 2
    ax.imshow(ground_truth_oh[:, 1:], aspect="auto", interpolation="none")
    ax.set_title("events")
    ax.set_yticks(range(0, len(prediction_oh), 100))
    ax.set_xticks(range(3))
    ax.set_xticklabels(categories[1:])
    ax.xaxis.tick_top()
    fig.tight_layout()
    fig.savefig("events_visualization.png")
    plt.show()


def results_for_file(file_path, labelled_file_path):
    event_generator.generate_events_for_file(file_path)
    event_path = "../Test_Mode/results/performance_results.csv"
    calculator.analyze_files(event_path, labelled_file_path)
    if show_plots:
        visualize_files(event_path, labelled_file_path)
    return "result"


if __name__ == "__main__":
    show_plots = True
    event_generator.print_info = False
    # event_generator.print_info = True
    for s in data_sets:
        print("-------------------------------- Dataset: " + s["description"])
        unlabelled_dir = s["path"] + "video_data/"
        labelled_dir = s["path"] + "video_data_labelled/"
        files = os.listdir(unlabelled_dir)
        filtered_files = [file for file in files if valid_filename(file)]
        for file in filtered_files:
            print(file)
            result = results_for_file(unlabelled_dir + file, labelled_dir + file)
