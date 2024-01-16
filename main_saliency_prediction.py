"""
Interface for saliency map prediction using ECT-SAL

This script takes input image path, text map path, and weight file path as command-line arguments.
It performs saliency map prediction and saves the output saliency map.

Usage:
python main_saliency_prediction.py --img_path path/to/input_image.jpg --weight_path path/to/weights.pth
                      --tmap path/to/text_map.jpg --output_path path/to/output_directory
"""

import argparse
import os

import cv2

from saliency_prediction import saliency_prediction_module


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Saliency Map Prediction')
    parser.add_argument('--img_path', type=str, default='test_images/test.jpg', help='path to input image')
    parser.add_argument('--weight_path', type=str, default='weights/ECT_SAL.pth',
                        help='path to weight file')
    parser.add_argument('--tmap', type=str, default='test_images/test_tmap.jpg', help='path to test tmap image')
    parser.add_argument('--output_path', type=str, default='results', help='path to output directory')
    args = parser.parse_args()

    # Check if the output directory exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Get the filename from the input image path
    filename = os.path.splitext(os.path.basename(args.img_path))[0]

    # Perform the saliency map prediction
    pred_saliency = saliency_prediction_module.saliency_map_prediction(args.img_path, args.tmap, args.weight_path)

    # Write the output saliency map
    output_filename = f'{filename}_saliencymap.png'
    output_path = os.path.join(args.output_path, output_filename)
    cv2.imwrite(output_path, pred_saliency)
    print(f"saliency map save in {output_path}")


if __name__ == '__main__':
    main()
