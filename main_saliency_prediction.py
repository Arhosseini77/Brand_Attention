from saliency_prediction import test_saliency_prediction

img_path = "test_images/test.jpg"
weight_path = "weights/Saliency_Map_Prediction_alpha.pth"
tmap = "test_images/test_tmap.jpg"

test_saliency_prediction.saliency_map_prediction(img_path,tmap ,weight_path,"./")