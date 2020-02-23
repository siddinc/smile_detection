import os
import cv2


FRAME_DIM = (640, 480)
ROI_DIM = (64, 64)
INPUT_DIM = (1, 64, 64, 1)
NO_OF_EPOCHS = 20
BATCH_SIZE = 64
DATASET_PATH = os.path.abspath('../datasets/smiles')
CASCADE_PATH = os.path.abspath('../datasets/haarcascade_frontalface_default.xml')
SAVE_MODEL_PATH = os.path.abspath('../models')
LOAD_MODEL_PATH = os.path.abspath('../models')
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE = 0.7
FONT_COLOR = (0, 255, 0)
FONT_THICKNESS = 2
SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 5
MIN_SIZE = (50, 50)
MAPPING = { 0: 'Not Smiling', 1: 'Smiling'}