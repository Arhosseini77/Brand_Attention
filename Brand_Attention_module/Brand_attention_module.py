import numpy as np
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
    print(score)


