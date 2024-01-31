import numpy as np
import torch
from torchvision import transforms

from saliency_prediction.model import ECT_SAL
from saliency_prediction.utils.data_process import preprocess_img, postprocess_img


def saliency_map_prediction(img_path, text_map_path, weight_path):
    """
    Predicts the saliency map of an image given its path and a text map path using the ECT_SAL model.

    Parameters:
    img_path (str): Path to the image file.
    text_map_path (str): Path to the text map file.
    weight_path (str): Path to the weights file for the ECT_SAL model.

    Returns:
    Image object of the predicted saliency map.

    This function loads a pre-trained ECT_SAL model, processes the input image and text map,
    and predicts the saliency map. The result is post-processed and returned as a PIL image.
    """

    # Set the computation device (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the ECT_SAL model
    model = ECT_SAL()
    model.load_state_dict(torch.load(weight_path), strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded...")

    # Preprocess the image and text map
    img = preprocess_img(img_path)
    tmap = preprocess_img(text_map_path)

    # Normalize and reshape the image and text map for the model
    img = np.array(img) / 255.
    tmap = np.array(tmap) / 255.
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    tmap = np.expand_dims(np.transpose(tmap, (2, 0, 1)), axis=0)

    # Convert to torch tensors
    img = torch.from_numpy(img)
    tmap = torch.from_numpy(tmap)

    # Move tensors to the computation device
    img = img.type(torch.cuda.FloatTensor).to(device)
    tmap = tmap.type(torch.cuda.FloatTensor).to(device)

    # Predict the saliency map
    pred_saliency = model(img, tmap)

    # Convert the prediction to a PIL Image
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    # Postprocess the saliency map
    pred_saliency = postprocess_img(pic, img_path)

    return pred_saliency


def saliency_map_prediction_brand(img_path , text_map_path):
    """
    Predicts the saliency map of an image given its path and a text map path,
    using the ECT_SAL model with predefined weights.

    Parameters:
    img_path (str): Path to the image file.
    text_map_path (str): Path to the text map file.

    Returns:
    Image object of the predicted saliency map.

    This function is similar to `saliency_map_prediction` but uses a predefined weight path
    for the ECT_SAL model. It is tailored for brand-attention applications
    """

    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Weight Path
    weight_path = "weights/ECT_SAL.pth"
    # load Model
    model = ECT_SAL()
    model.load_state_dict(torch.load(weight_path) , strict=False)
    model = model.to(device)
    model.eval()

    img = preprocess_img(img_path)
    tmap = preprocess_img(text_map_path)

    img = np.array(img) / 255.
    tmap = np.array(tmap) / 255.

    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    tmap = np.expand_dims(np.transpose(tmap, (2, 0, 1)), axis=0)

    # Convert to torch tensors
    img = torch.from_numpy(img)
    tmap = torch.from_numpy(tmap)

    # Move tensors to the computation device
    img = img.type(torch.cuda.FloatTensor).to(device)
    tmap = tmap.type(torch.cuda.FloatTensor).to(device)

    # Predict the saliency map
    pred_saliency = model(img, tmap)

    # Convert the prediction to a PIL Image
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    # Postprocess the saliency map
    pred_saliency = postprocess_img(pic, img_path)

    return pred_saliency
