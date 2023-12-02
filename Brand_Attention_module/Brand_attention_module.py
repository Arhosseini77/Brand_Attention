import numpy as np
import cv2
from saliency_prediction.test_saliency_prediction import saliency_map_prediction_brand
from logo_detection.yolov8_logo import yolov8_logo_detection

def calculate_sum_of_probabilities(saliency_map, bounding_boxes):
    if bounding_boxes == "none":
        return "none"
    else:
        # Apply threshold to saliency map
        saliency_map[saliency_map < 100] = 0
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


# Initialize global variables
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial x and y coordinates
bboxes = []      # List to store bounding box coordinates

# Mouse callback function for drawing bboxes
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, img, bboxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
        bboxes.append([ix, iy, x, y])

# Function to draw rectangles on the image
def draw_rectangles(image, bboxes):
    for bbox in bboxes:
        xmin, ymin, xmax, ymax = map(int, bbox)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    return image

# Main function for brand attention calculation
def brand_attention_calc2(img_path, tmap_path):
    global img, bboxes, drawing

    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading image from {img_path}")
        return

    # Load detected bboxes - Implement your function here
    detected_bboxes = yolov8_logo_detection("weights/Logo_Detection_Yolov8.pt", img_path, save_result=False)

    # Draw rectangles on the image
    img_with_boxes = draw_rectangles(img.copy(), detected_bboxes)

    # Show image with detected bboxes
    cv2.imshow('Image', img_with_boxes)
    print("Check the image window. Press '1' if OK, '2' to draw boxes, then press 'Enter'.")

    key = cv2.waitKey(0)

    if key == ord('2'):
        cv2.setMouseCallback('Image', draw_bbox)
        print("Draw boxes. Press 'Enter' key in the image window when done.")
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
