import os
import cv2
from ultralytics import YOLO


def yolov8_logo_detection(model_path, image_path , save_result = False):
    # load Model
    model = YOLO(model_path)

    # get bbox
    results = model(image_path)
    results = (results[0].boxes).xyxy
    bboxes = []
    for box in results:
        bboxes.append(box.cpu().tolist())

    if save_result:
        image = cv2.imread(image_path)
        filename = image_path.split("/")[-1].split(".")[0]
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Draw a rectangle around the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0),
                          2)  # (0, 255, 0) is the color (green), and 2 is the thickness

        # Save image with bounding boxes
        os.makedirs("./results", exist_ok=True)
        cv2.imwrite(f'results/{filename}_detected_logo.png', image)
    return bboxes
