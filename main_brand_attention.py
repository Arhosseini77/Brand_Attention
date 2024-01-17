"""
 Interface for Brand-Attention Score Calculation.

 Usage:
 python main_brand_attention.py --img_path path/to/input_image.jpg --tmap path/to/text_map.jpg

 Arguments:
 --img_path : Path to the input image file.
 --tmap     : Path to the text map image file.
 """

import argparse

from brand_attention_module.brand_attention_module import brand_attention_calc2


def main():
    parser = argparse.ArgumentParser(description='Brand Attention Score Calculation')
    parser.add_argument('--img_path', type=str, default='test_images/test.jpg', help='path to input image')
    parser.add_argument('--tmap', type=str, default='test_images/test_tmap.jpg', help='path to text map image')
    args = parser.parse_args()

    # Calculate brand change score
    brand_change_score = brand_attention_calc2(args.img_path, args.tmap)

    print(f"Brand Attention Score: {brand_change_score}")


if __name__ == '__main__':
    main()

