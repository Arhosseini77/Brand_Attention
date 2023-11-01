import os
import cv2
import numpy as np
import math
import csv
from skimage import img_as_float
import torch as t
from skimage import exposure
from scipy.ndimage import zoom
from skimage.transform import resize
import argparse  
from tqdm import tqdm

def normalize(x, method='standard', axis=None):
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def loss_similarity(pred_map, gt_map):
    # Normalize and scale the ground truth map
    gt_map = (gt_map - np.min(gt_map)) / (np.max(gt_map) - np.min(gt_map))
    gt_map = gt_map / np.sum(gt_map)

    # Normalize and scale the predicted map
    pred_map = (pred_map - np.min(pred_map)) / (np.max(pred_map) - np.min(pred_map))
    pred_map = pred_map / np.sum(pred_map)

    # Calculate the element-wise minimum
    diff = np.minimum(gt_map, pred_map)

    # Calculate the similarity score
    score = np.sum(diff)

    return score

def cc(s_map, gt):
    s_map_norm = (s_map - np.mean(s_map)) / np.std(s_map)
    gt_norm = (gt - np.mean(gt)) / np.std(gt)
    a = s_map_norm
    b = gt_norm
    r = (a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
    return r

def kldiv(s_map,gt):
    eps = 2.2204e-16
    s_map = s_map/np.sum(s_map)
    gt = gt/np.sum(gt)
    div = np.sum(np.multiply(gt, np.log(eps + np.divide(gt,s_map+eps))))
    return div


def AUC_Judd(saliency_map, fixation_map, jitter=True):
    saliency_map = np.array(saliency_map, copy=False)
    fixation_map = np.array(fixation_map, copy=False) > 0.5
    # If there are no fixations to predict, return NaN
    if not np.any(fixation_map):
        print('No fixations to predict')
        return np.nan
    # Make the saliency_map the size of the fixation_map
    if saliency_map.shape != fixation_map.shape:
        saliency_map = resize(saliency_map, fixation_map.shape, order=3, mode='nearest')
    # Jitter the saliency map slightly to disrupt ties of the same saliency value
    if jitter:
        # Generate random numbers in the same shape as saliency_map as float64
        random_values = np.random.rand(*saliency_map.shape).astype(np.float64)
        saliency_map = saliency_map.astype(np.float64) + random_values * 1e-7
    # Normalize saliency map to have values between [0,1]
    saliency_map = normalize(saliency_map, method='range')

    S = saliency_map.ravel()
    F = fixation_map.ravel()
    S_fix = S[F]  # Saliency map values at fixation locations
    n_fix = len(S_fix)
    n_pixels = len(S)
    # Calculate AUC
    thresholds = sorted(S_fix, reverse=True)
    tp = np.zeros(len(thresholds) + 2)
    fp = np.zeros(len(thresholds) + 2)
    tp[0] = 0
    tp[-1] = 1
    fp[0] = 0
    fp[-1] = 1
    for k, thresh in enumerate(thresholds):
        above_th = np.sum(S >= thresh)  # Total number of saliency map values above threshold
        tp[k + 1] = (k + 1) / float(n_fix)  # Ratio of saliency map values at fixation locations above threshold
        fp[k + 1] = (above_th - k - 1) / float(n_pixels - n_fix)  # Ratio of other saliency map values above threshold
    return np.trapz(tp, fp)  # y, x

def loss_NSS(pred_map, fix_map):
    '''ground truth here is a fixation map'''

    pred_map_ = (pred_map - np.mean(pred_map)) / np.std(pred_map)

    # Convert the fixation map to a binary mask
    fix_map_binary = fix_map > 0

    score = np.mean(pred_map_[fix_map_binary])
    return score


def main(args):
    
    val_saliency_dir = args.saliency_dir
    val_fixation_dir = args.fixation_dir
    output_dir = args.output_dir
    

    
    val_saliency_files = os.listdir(val_saliency_dir)
    output_files = os.listdir(output_dir)
    val_fixation_files = os.listdir(val_fixation_dir)

    
    results = []

    for val_file in val_saliency_files:
        if val_file.endswith('.jpg'):
            img_number = val_file.split('_')[0]
            img_number = int(img_number)

            output_file = f"{img_number}.jpg"
            if output_file in output_files:

                val_image = cv2.imread(os.path.join(val_saliency_dir, val_file), cv2.IMREAD_GRAYSCALE)
                output_image = cv2.imread(os.path.join(output_dir, output_file), cv2.IMREAD_GRAYSCALE)
                output_image = cv2.resize(output_image, (720, 720))

                fixation_file = f"{img_number}_fixPts.jpg"
                if fixation_file in val_fixation_files:
                    fixation_image = cv2.imread(os.path.join(val_fixation_dir, fixation_file), cv2.IMREAD_GRAYSCALE)

                    # Calculate the similarity score
                    similarity_score = loss_similarity(output_image, val_image)

                    # Calculate the cc value
                    cc_value = cc(output_image, val_image)

                    # Calculate the kldiv value
                    kldiv_value = kldiv(output_image, val_image)

                    # Calculate the AUC_Judd value
                    auc_judd_value = AUC_Judd(output_image, fixation_image)

                    # Calculate the NSS value
                    nss_value = loss_NSS(output_image, fixation_image)

                    results.append([img_number, similarity_score, cc_value, kldiv_value, auc_judd_value, nss_value])

    results.sort(key=lambda x: x[0])
        
    csv_file = args.output_file
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Number', 'Similarity Score', 'CC Value', 'KLDiv Value', 'AUC_Judd Value', 'NSS Value'])
        writer.writerows(results)

    print(f"Metrics (Similarity, CC, KLDiv, AUC_Judd, NSS) have been saved to {csv_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('saliency_dir', type=str, help='Path to the saliency maps directory')
    parser.add_argument('fixation_dir', type=str, help='Path to the fixation maps directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory where the results will be saved')
    parser.add_argument('output_file', type=str, help='Path to the output CSV file')
    args = parser.parse_args()
    main(args)
