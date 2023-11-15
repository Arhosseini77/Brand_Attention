import argparse

from Brand_Attention_module.Brand_attention_module import brand_attention_calc

def main():
    parser = argparse.ArgumentParser(description='Brand Change Score Calculation')
    parser.add_argument('--img_path', type=str, default='test_images/test.jpg', help='path to input image')
    parser.add_argument('--tmap', type=str, default='test_images/test_tmap.jpg', help='path to text map image')
    args = parser.parse_args()

    # Calculate brand change score
    brand_change_score = brand_attention_calc(args.img_path, args.tmap)

    print(f"Brand Change Score: {brand_change_score}")


if __name__ == '__main__':
    main()

