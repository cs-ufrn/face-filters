# face-filters cs-ufrn
A web + computer vision project which implements a facial tracker along with some real-time filters and effects.

## Environment/dependencies
Python (>= 3.6), along with the following packages:
- numpy
- Flask
- OpenCV
- dlib

Additionally, the facial features recognition model, "shape_predictor_68_face_landmarks.dat", must be present in the root directory of the projet, and can be downloaded and uncompressed from the link below:

[Download facial landmarks model](https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2)

## Running
The web interface runs on Flask, and can be (currently, for demonstration purposes) run simply as:
```bash
python web.py
```
This will run the web server locally on debug mode, at the address 0.0.0.0:8080
