import cv2
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from constants import (
    FRAME_DIM,
    SCALE_FACTOR,
    MIN_NEIGHBORS,
    MIN_SIZE,
    ROI_DIM,
    INPUT_DIM,
    FONT,
    FONT_SCALE,
    FONT_COLOR,
    FONT_THICKNESS
)


def load_dataset(dataset_path):
    data = []
    labels = []

    print("[INFO] loading dataset...")

    for image_path in sorted(list(paths.list_images(dataset_path))):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        np_image = img_to_array(gray_image)
        data.append(np_image)

        label = image_path.split('/')[-3]
        label = 'smiling' if label == 'positives' else 'not_smiling'
        labels.append(label)

    data = np.array(data, dtype='uint8')
    labels = np.array(labels)
    return (data, labels)


def preprocess_dataset(data, labels):
    classes = len(set(labels))

    print("[INFO] preprocessing dataset...")

    data = data / data.max()
    
    le = LabelEncoder().fit(labels)
    labels = to_categorical(le.transform(labels), classes)

    class_totals = np.sum(labels, axis=0)
    class_weight = class_totals.max() / class_totals
    
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20)
    return (x_train, x_test, y_train, y_test, class_weight, classes)


def preprocess_frame(frame):
    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, FRAME_DIM)
    frame_clone = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return (frame_clone, gray_frame)


def detect_face(detector, gray_image):
    rois = []

    faces = detector.detectMultiScale(
        gray_image,
        scaleFactor=SCALE_FACTOR,
        minNeighbors=MIN_NEIGHBORS, 
        minSize=MIN_SIZE,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    for (fX, fY, fW, fH) in faces:
        roi = gray_image[fY : fY+fH, fX : fX+fW]
        roi = cv2.resize(roi, ROI_DIM)
        np_img = np.reshape(roi, INPUT_DIM)
        np_img = np_img / np_img.max()
        rois.append({
            "roi": np_img,
            "fX": fX,
            "fY": fY,
            "fW": fW,
            "fH": fH,
        })
    return rois


def display_roi(frame_clone, label, score, roi):
    cv2.putText(
        frame_clone, 
        "{}: {}%".format(label, int(score)),
        (roi["fX"], roi["fY"] - 10),
        FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS
    )
    cv2.rectangle(
        frame_clone,
        (roi["fX"], roi["fY"]),
        (roi["fX"] + roi["fW"], roi["fY"] + roi["fH"]),
        FONT_COLOR, FONT_THICKNESS
    )


def plot_metrics(r):
    plt.subplot(121)
    plt.plot(r.history['loss'], label='loss')
    plt.plot(r.history['val_loss'], label='val_loss')
    plt.legend()

    plt.subplot(122)
    plt.plot(r.history['accuracy'], label='acc')
    plt.plot(r.history['val_accuracy'], label='val_acc')
    plt.legend()

    plt.show()