# Machine Learning Final Project

This project presents a gesture classifier developed with a custom machine learning package and integrated with the reveal.js slideshow framework. It provides users with the ability to control the slideshow using gestures.

This repository contains the main project code as well as the [Teaser Video](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/final-submission/-/blob/main/Teaser/MachineLearning_WS2223_Team7_Teaser.mp4) and the [presentation slides](https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/final-submission/-/blob/main/Teaser/ML_Presentation.pdf).

## Set-up
1. Clone the project on your local machine (we recommend to use `git clone --depth 1 https://gitlab2.informatik.uni-wuerzburg.de/hci/teaching/courses/machine-learning/student-submissions/ws22/Team-7/final-submission.git` to not clone history including large video files)
2. Create a new conda environment with python version 3.7
3. Activate the environment, navigate to the main directory of the project and install the requirements via
   `pip install -r requirements.txt`

## Usage
### Prediction mode for Reveal.js Slideshow:
1. Run `/Prediction_Mode/slideshow.py` script.
2. Click on the IP address `http://127.0.0.1:8000` in the python console to open the slideshow in the browser
3. Focus the slideshow. 

#### Usable Gestures:
- Swipe Left -> One slide to the left
- Swipe Right -> One slide to the right
- Swipe Up -> One slide upwards
- Swipe Down -> One slide downwards
- Rotate Right -> Rotates all rotatable elements to the right
- Rotate Right -> Rotates all rotatable elements to the left
- Spread -> Increases the slide size
- Pinch -> Decreases the slide size
- Flip Table -> Opens / Closes the overview mode
- Spin -> If the current slide contains a rotatable picture, it spins by 360° | If the current slide contains a video, it changes the video speed from 1x to 4x and vice versa
- Point -> Starts / Stops a video

### Prediction mode for O9 Tetris:
1. Run `/Game/tetris.py` script.

#### Usable Gestures:
- Swipe Left -> Tetris tile to the left
- Swipe Right -> Tetris tile to the right
- Swipe Down -> Increases the downward speed of a Tetris tile
- Rotate Left -> Rotates the Tetris tile to the left
- Rotate Right -> Rotates the Tetris tile to the right
- Flip Table -> Changes the Game status to Play / Pause
- Spin -> Rotates the Tetris grid by 180°

### Test mode:
1. Place the csv file for which events should be generated in `/Test_Mode/test_data/`
2. Make sure there are no other files in the `test_data` directory
3. Run `/Test_Mode/event_generator.py`
4. The generated result file will be saved to `/Test_Mode/results/performance_results.csv`


## Troubleshooting

* **OpenCV-Python Package:**
  If the used methods from CV2 cannot be found and are thus highlighted, it is necessary to [add the CV2 path to the interpreter paths](https://github.com/opencv/opencv/issues/20997#issuecomment-1328068006).

## Authors and acknowledgment
Nowak Micha, Roth Marcel, Friese Jan-Philipp

## License
MIT License
