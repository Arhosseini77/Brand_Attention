"""
Command-line interface for YOLOv8-based Logo Detection.

This script performs logo detection using the YOLOv8 model on an input image.
It takes the path to the YOLOv8 model, the input image, and an optional flag to save the result image.

Usage:
python main_detection_yolov8.py --model path/to/model.pt --image path/to/image --save-result

Arguments:
--model          : Path to the YOLOv8 model file.
--image          : Path to the input image file.
--save-result    : Flag to save the result image with bounding boxes.
"""

import argparse

from logo_detection.logo_detection_module import yolov8_logo_detection


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Logo Detection")
    parser.add_argument("--model", type=str, default='weights/Logo_Detection_Yolov8.pt'
                        , help="Path to YOLOv8 trained model")
    parser.add_argument("--image", type=str, default='test_images/test.jpg', help="Path to input image")
    parser.add_argument("--save-result", action="store_true", help="Save the result image with bounding boxes")

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    save_result = args.save_result

    # Perform logo detection and get bounding boxes
    bboxes = yolov8_logo_detection(model_path, image_path, save_result)
    print("bounding-boxes list :")
    print(bboxes)

    if save_result:
        print(f"Bounding boxes saved to ./results/{image_path.split('/')[-1].split('.')[0]}_detected_logo.png")


if __name__ == "__main__":
    main()