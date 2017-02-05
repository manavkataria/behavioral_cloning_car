# utils.py
import cv2
import numpy as np

from settings import (HEIGHT, WIDTH, DEPTH,
                      ROI_bbox,
                      DISPLAY_IMAGES)


def preprocess_for_autonomy(raw_image):
    image = preprocess_image(raw_image)
    display_images(image)
    return image


def print_predictions(model, X_train, y_train):
    predictions = model.predict_on_batch(X_train)
    for i in range(len(predictions)):
        print("Prediction[{:02d}]: {:>3d} <= {:>3d} - {:>3d}".format(
              i,
              int(predictions[i][0] - y_train[i]),
              int(predictions[i][0]),
              int(y_train[i])))


def cut_ROI_bbox(image_data):
    w = image_data.shape[1]
    h = image_data.shape[0]
    x1 = int(w * ROI_bbox[0])
    x2 = int(w * (1 - ROI_bbox[2]))
    y1 = int(h * ROI_bbox[1])
    y2 = int(h * (1 - ROI_bbox[3]))
    ROI_data = image_data[y1:y2, x1:x2]
    return ROI_data


def rgb_to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_grayscale(imgray):
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + (((imgray - grayscale_min) * (b - a)) / (grayscale_max - grayscale_min))


def preprocess_image(image):
    """ Resize and Crop Image """
    gray = rgb_to_grayscale(image)
    cropped = cut_ROI_bbox(gray)
    resized = cv2.resize(cropped, (WIDTH, HEIGHT))
    normalized = normalize_grayscale(resized)
    reshaped = normalized.reshape(HEIGHT, WIDTH, DEPTH)
    return reshaped


def display_images(image_features, message=None, delay=500):
    if not DISPLAY_IMAGES: return
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (255, 255, 255)
    FONT_THICKNESS = 1
    # FONT_SCALE = 4

    if image_features.ndim == 3:
        image_features = [image_features]

    height, width, depth = image_features[0].shape

    for image in image_features:
        image = np.copy(image)  # Avoid Overwriting Original Image
        if message:
            text_position = (int(width * 0.05), int(height * 0.95))
            cv2.putText(image, message, text_position, font, FONT_THICKNESS, WHITE)
        cv2.imshow(message, image)
        cv2.waitKey(delay)
