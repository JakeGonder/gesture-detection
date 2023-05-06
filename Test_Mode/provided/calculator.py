import argparse
import pandas as pd

AVAILABLE_GESTURES = ["idle", "swipe_left", "swipe_right", "rotate"]


def read_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--events_csv",
                        help="CSV file containing a column 'prediction' with a prediction for each frame",
                        required=True)
    parser.add_argument("--ground_truth_csv",
                        help="CSV file containing a column 'ground_truth' with the correct gesture for each frame (may be the same as 'events_csv')",
                        required=True)

    args = parser.parse_known_args()[0]
    return args


def read_data_callable(event_path, ground_truth_path):
    events = pd.read_csv(event_path)
    ground_truth = pd.read_csv(ground_truth_path)
    # necessary since our rotate rights are not named rotate
    ground_truth["ground_truth"] = ground_truth["ground_truth"].replace("rotate_right", "rotate")

    assert len(events) == len(ground_truth), "events and ground truth CSV are not of the same length!"

    # drop rows where no ground truth is given
    rows_to_drop = ground_truth["ground_truth"].isna().values
    ground_truth = ground_truth[~rows_to_drop]
    events = events[~rows_to_drop]

    return events, ground_truth

def read_data(args):
    events = pd.read_csv(args.events_csv)
    ground_truth = pd.read_csv(args.ground_truth_csv)

    assert len(events) == len(ground_truth), "events and ground truth CSV are not of the same length!"

    # drop rows where no ground truth is given
    rows_to_drop = ground_truth["ground_truth"].isna().values
    ground_truth = ground_truth[~rows_to_drop]
    events = events[~rows_to_drop]

    return events, ground_truth


def count_individual_gestures(ground_truth):
    return (ground_truth.iloc[:-1] != ground_truth.shift(-1).iloc[:-1]).sum() // 2


def calculate_scores(events_df, ground_truth_df, bonus=10, malus=0.2):
    events = events_df["events"]
    ground_truth = ground_truth_df["ground_truth"]

    assert len(events) == len(ground_truth), "Error: the CSV files differ in length!"

    num_frames = len(ground_truth)
    num_total_gestures = count_individual_gestures(ground_truth)

    last_frame_gesture = "idle"
    current_gesture_detected = False
    mistakes = []
    gesture_hits = 0

    marks = 0

    for frame_idx in range(num_frames):
        current_gesture = ground_truth.iloc[frame_idx]
        current_event = events.iloc[frame_idx]

        event_fired = current_event != "idle"
        gesture_in_progress = current_gesture != "idle"

        fired_wrong_event = gesture_in_progress and \
                            event_fired and \
                            current_event != current_gesture

        fired_event_but_no_gesture_in_progress = event_fired and not gesture_in_progress
        fired_event_more_than_once = event_fired and current_gesture_detected

        if last_frame_gesture != current_gesture:
            gesture_ended_but_no_event_fired = not gesture_in_progress and not current_gesture_detected
            current_gesture_detected = False
        else:
            gesture_ended_but_no_event_fired = False

        if fired_event_but_no_gesture_in_progress or \
                fired_event_more_than_once or \
                fired_wrong_event or \
                gesture_ended_but_no_event_fired:
            marks -= malus
            # print(f"malus {fired_event_but_no_gesture_in_progress} {fired_event_more_than_once} {fired_wrong_event} {gesture_ended_but_no_event_fired}")
        elif gesture_in_progress and current_event == current_gesture:
            marks += 1
            gesture_hits += 1
            # print("correct")
            current_gesture_detected = True
        if fired_event_but_no_gesture_in_progress:
            mistakes.append("detection_without_gesture")
        if fired_event_more_than_once:
            mistakes.append("double_detection_same_gesture")
        if fired_wrong_event:
            mistakes.append("detected_wrong_gesture")
        if gesture_ended_but_no_event_fired:
            mistakes.append("undetected_gesture")


        last_frame_gesture = current_gesture

    total_points = num_total_gestures
    score = max((marks / total_points) * 100 + bonus, 0)

    # print("marks: %d/%d" % (marks, total_points))
    print(f"marks:{marks} / {total_points} Gesture Hits: {gesture_hits} Mistakes: " + " ".join(mistakes))
    print("Score: %.2f%%" % score)

def analyze_files(event_path, ground_truth_path):
    events, ground_truth = read_data_callable(event_path, ground_truth_path)
    calculate_scores(events, ground_truth)


if __name__ == "__main__":
    args = read_command_line_arguments()
    events, ground_truth = read_data(args)
    calculate_scores(events, ground_truth)
