"""This module contains functions for calculating brand attention in packaging and advertisement images. It includes
functionalities for merge logo detection result , saliency map prediction result, and processing these results to
compute a brand attention score. """

import cv2
import numpy as np

from logo_detection.logo_detection_module import yolov8_logo_detection
from saliency_prediction.saliency_prediction_module import saliency_map_prediction_brand


def calculate_sum_of_probabilities(saliency_map, bounding_boxes):
    if bounding_boxes == "none":
        return "none"
    else:
        # Apply threshold to saliency map
        saliency_map[saliency_map < 80] = 0
        # Normalize the saliency map
        normalized_map = saliency_map / np.sum(saliency_map)

        # Initialize the sum of probabilities
        sum_of_probabilities = 0.0

        # Iterate through the bounding boxes
        for box in bounding_boxes:
            try:
                xmin, ymin, xmax, ymax = box
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                # Iterate through the pixels within the bounding box
                for y in range(ymin, ymax + 1):
                    for x in range(xmin, xmax + 1):
                        # Accumulate the normalized probability of each pixel
                        sum_of_probabilities += normalized_map[y, x]
            except:
                print("No brand found")
        return sum_of_probabilities


def brand_attention_calc(img_path , tmap_path ):
    pred_saliency = saliency_map_prediction_brand(img_path , tmap_path)
    bboxes = yolov8_logo_detection("weights/Logo_Detection_Yolov8.pt", img_path, save_result=False)
    score = calculate_sum_of_probabilities(pred_saliency, bboxes)
    return score


# Function to resize image while maintaining aspect ratio
def resize_image_aspect_ratio(image, width=None):
    (h, w) = image.shape[:2]

    if width is None:
        return image, 1  # Return the same image and scale factor 1

    r = width / float(w)
    dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized, r


# Function to draw rectangles on the image
def draw_rectangles(image, bboxes):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image


# Mouse callback function for drawing bboxes
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, img, bboxes, resize_scale

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        # Scale bbox coordinates back to original image size
        scaled_bbox = [ix / resize_scale, iy / resize_scale, x / resize_scale, y / resize_scale]
        bboxes.append(scaled_bbox)


# Main function for brand attention calculation
def brand_attention_calc2(img_path, tmap_path):
    global img, bboxes, drawing, resize_scale

    # Load image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Error loading image from {img_path}")
        return

    # Resize image for display
    img, resize_scale = resize_image_aspect_ratio(original_img, width=720)

    # Scale detected bboxes according to the resized image
    detected_bboxes = yolov8_logo_detection("weights/Logo_Detection_Yolov8.pt", img_path, save_result=False)
    scaled_detected_bboxes = [[x * resize_scale, y * resize_scale, w * resize_scale, h * resize_scale] for [x, y, w, h] in detected_bboxes]

    # Draw rectangles on the resized image
    img_with_boxes = draw_rectangles(img.copy(), scaled_detected_bboxes)
    cv2.imshow('Image', img_with_boxes)
    print("Check the image window. Press '1' if OK, '2' to draw boxes, then press 'Enter'.")

    key = cv2.waitKey(0)

    if key == ord('2'):
        bboxes = []  # Reset bboxes
        cv2.setMouseCallback('Image', draw_bbox)
        print("Draw boxes. Press 'Enter' key in the console when done.")
        while True:
            cv2.imshow('Image', img)
            if cv2.waitKey(1) & 0xFF == 13:  # Enter key to finish
                break
    elif key == ord('1'):
        bboxes = detected_bboxes

    cv2.destroyAllWindows()

    # Continue with the brand attention calculation
    pred_saliency = saliency_map_prediction_brand(img_path, tmap_path)
    score = calculate_sum_of_probabilities(pred_saliency, bboxes)
    return score


def object_attention_calc(img_path, tmap_path):
    global img, bboxes, drawing, resize_scale

    # Load image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Error loading image from {img_path}")
        return

    # Resize image for display
    img, resize_scale = resize_image_aspect_ratio(original_img, width=720)

    # Display the image for the user to draw bounding boxes
    cv2.imshow('Image', img)
    print("Draw boxes on the image. Press 'Enter' key when done.")

    # Initialize bounding boxes list and set the callback for mouse events
    bboxes = []
    cv2.setMouseCallback('Image', draw_bbox)

    # Loop to keep the window open until 'Enter' key is pressed
    while True:
        cv2.imshow('Image', img)
        # Enter key to finish
        if cv2.waitKey(1) & 0xFF == 13:
            break

    cv2.destroyAllWindows()

    # Continue with the object attention calculation
    pred_saliency = saliency_map_prediction_brand(img_path, tmap_path)
    score = calculate_sum_of_probabilities(pred_saliency, bboxes)
    return score