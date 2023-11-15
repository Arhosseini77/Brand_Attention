import torch
import numpy as np
from torchvision import transforms
from saliency_prediction.utils.data_process import preprocess_img, postprocess_img
from saliency_prediction.model import ECT_SAL


def saliency_map_prediction(img_path , text_map_path , weight_path ):

    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load Model
    model = ECT_SAL()
    model.load_state_dict(torch.load(weight_path), strict=False)
    model = model.to(device)
    model.eval()
    print("Model loaded...")

    img = preprocess_img(img_path)
    name = img_path.split('/')[-1].split('.')[0]
    tmap = preprocess_img(text_map_path)

    img = np.array(img) / 255.
    tmap = np.array(tmap) / 255.

    img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    tmap = np.expand_dims(np.transpose(tmap, (2, 0, 1)), axis=0)

    img = torch.from_numpy(img)
    tmap = torch.from_numpy(tmap)

    img = img.type(torch.cuda.FloatTensor).to(device)
    tmap = tmap.type(torch.cuda.FloatTensor).to(device)

    pred_saliency = model(img, tmap)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())
    pred_saliency = postprocess_img(pic, img_path)

    return pred_saliency



def saliency_map_prediction_brand(img_path , text_map_path):

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

    img = torch.from_numpy(img)
    tmap = torch.from_numpy(tmap)

    img = img.type(torch.cuda.FloatTensor).to(device)
    tmap = tmap.type(torch.cuda.FloatTensor).to(device)

    pred_saliency = model(img, tmap)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())
    pred_saliency = postprocess_img(pic, img_path)

    return pred_saliency
