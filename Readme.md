# Real-Time Smile Detection
The goal of this project is to build and train a model which is able to classify a smiling and a non smiling face in real-time.

## Tech used:
- TensorFlow 2.0.0
- OpenCV 3.1.0
- Python 3.5.6

## Dataset:
- SMILES Dataset used for training and testing
- 13165 images of faces that are either smiling or non-smiling
- All images are grayscale with dimensions 64 x 64 pixels
- Number of classes: 2

## Trained Models:
`model1.h5` has the following accuracy metrics:
  - Training accuracy = 91.78%
  - Validation accuracy = 90.58%
> `model1.h5` was trained for 20 epochs with a batch size of 64

## Instructions to run:
- Using `anaconda`:
  - Run `conda create --name <env_name> --file recog.yml`
  - Run `conda activate <env_name>`
- Using `pip`:
  - Run `pip install -r requirements.txt`
- `cd` to `src`
- Run `python main.py`