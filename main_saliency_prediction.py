import argparse
import os

import cv2

from saliency_prediction import test_saliency_prediction


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Saliency Map Prediction')
    parser.add_argument('--img_path', type=str, default='test_images/test.jpg', help='path to input image')
    parser.add_argument('--weight_path', type=str, default='weights/Saliency_Map_Prediction_alpha.pth',
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
    pred_saliency = test_saliency_prediction.saliency_map_prediction(args.img_path, args.tmap, args.weight_path)

    # Write the output saliency map
    output_filename = f'{filename}_saliencymap.png'
    output_path = os.path.join(args.output_path, output_filename)
    cv2.imwrite(output_path, pred_saliency)
    print(f"saliency map save in {output_path}" )


if __name__ == '__main__':
    main()
