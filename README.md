# Brand Attention

## Installation

````
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
````

## Brand-Logo-Detection

### Description

This module focuses on detecting brand logos in images using the YOLOv8 model. It utilizes two datasets for training: [FoodLogoDet-1500](https://github.com/hq03/FoodLogoDet-1500-Dataset) and [LogoDet-3K](https://github.com/Wangjing1551/LogoDet-3K-Dataset).

### Inference

You can use the following command to run the brand logo detection code:

```shell
python main_detection_yolov8.py --model="weights/Logo_Detection_Yolov8.pt" --image="test_images/test.jpg" --save-result
```
* If you want to visualize the detection results, include the --save-result flag in the command.

### Result

