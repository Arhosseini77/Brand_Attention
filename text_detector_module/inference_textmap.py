"""
The script is set up to process each .png image in the specified input directory,
detect text regions using the DBNet model,
and then save the modified images where the detected text areas are highlighted to the specified output directory.
"""

import math
import cv2
import os.path as osp
import glob
import argparse
import numpy as np
from shapely.geometry import Polygon
from keras_resnet.models import ResNet50
from keras import layers, models
import tensorflow as tf
import pyclipper
import keras.backend as K

def parse_args():
    """
    Parse command line arguments for the textmap generator.

    Returns:
        Namespace: Parsed arguments with 'input_dir' and 'output_dir'.
    """
    parser = argparse.ArgumentParser(description='Textmap generator')
    parser.add_argument('--input_dir', type=str, help='Input directory path')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    return parser.parse_args()

def balanced_crossentropy_loss(args, negative_ratio=3., scale=5.):
    pred, gt, mask = args
    pred = pred[..., 0]
    positive_mask = (gt * mask)
    negative_mask = ((1 - gt) * mask)
    positive_count = tf.reduce_sum(positive_mask)
    negative_count = tf.reduce_min([tf.reduce_sum(negative_mask), positive_count * negative_ratio])
    loss = K.binary_crossentropy(gt, pred)
    positive_loss = loss * positive_mask
    negative_loss = loss * negative_mask
    negative_loss, _ = tf.nn.top_k(tf.reshape(negative_loss, (-1,)), tf.cast(negative_count, tf.int32))

    balanced_loss = (tf.reduce_sum(positive_loss) + tf.reduce_sum(negative_loss)) / (
            positive_count + negative_count + 1e-6)
    balanced_loss = balanced_loss * scale
    return balanced_loss, loss


def dice_loss(args):
    """

    Args:
        pred: (b, h, w, 1)
        gt: (b, h, w)
        mask: (b, h, w)
        weights: (b, h, w)
    Returns:

    """
    pred, gt, mask, weights = args
    pred = pred[..., 0]
    weights = (weights - tf.reduce_min(weights)) / (tf.reduce_max(weights) - tf.reduce_min(weights)) + 1.
    mask = mask * weights
    intersection = tf.reduce_sum(pred * gt * mask)
    union = tf.reduce_sum(pred * mask) + tf.reduce_sum(gt * mask) + 1e-6
    loss = 1 - 2.0 * intersection / union
    return loss


def l1_loss(args, scale=10.):
    pred, gt, mask = args
    pred = pred[..., 0]
    mask_sum = tf.reduce_sum(mask)
    loss = K.switch(mask_sum > 0, tf.reduce_sum(tf.abs(pred - gt) * mask) / mask_sum, tf.constant(0.))
    loss = loss * scale
    return loss


def db_loss(args):
    binary, thresh_binary, gt, mask, thresh, thresh_map, thresh_mask = args
    l1_loss_ = l1_loss([thresh, thresh_map, thresh_mask])
    balanced_ce_loss_, dice_loss_weights = balanced_crossentropy_loss([binary, gt, mask])
    dice_loss_ = dice_loss([thresh_binary, gt, mask, dice_loss_weights])
    return l1_loss_ + balanced_ce_loss_ + dice_loss_



def dbnet(input_size=640, k=50):
    """
    Construct DBNet for text detection.

    Args:
        input_size (int): Size of the input image.
        k (float): Scaling factor for the binary map.

    Returns:
        Tuple[models.Model, models.Model]: Training and prediction models.
    """
    image_input = layers.Input(shape=(None, None, 3))
    gt_input = layers.Input(shape=(input_size, input_size))
    mask_input = layers.Input(shape=(input_size, input_size))
    thresh_input = layers.Input(shape=(input_size, input_size))
    thresh_mask_input = layers.Input(shape=(input_size, input_size))
    backbone = ResNet50(inputs=image_input, include_top=False, freeze_bn=True)
    C2, C3, C4, C5 = backbone.outputs
    in2 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in2')(C2)
    in3 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in3')(C3)
    in4 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in4')(C4)
    in5 = layers.Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal', name='in5')(C5)

    # 1 / 32 * 8 = 1 / 4
    P5 = layers.UpSampling2D(size=(8, 8))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(in5))
    # 1 / 16 * 4 = 1 / 4
    out4 = layers.Add()([in4, layers.UpSampling2D(size=(2, 2))(in5)])
    P4 = layers.UpSampling2D(size=(4, 4))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out4))
    # 1 / 8 * 2 = 1 / 4
    out3 = layers.Add()([in3, layers.UpSampling2D(size=(2, 2))(out4)])
    P3 = layers.UpSampling2D(size=(2, 2))(
        layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(out3))
    # 1 / 4
    P2 = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(
        layers.Add()([in2, layers.UpSampling2D(size=(2, 2))(out3)]))
    # (b, /4, /4, 256)
    fuse = layers.Concatenate()([P2, P3, P4, P5])

    # probability map
    p = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(p)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                               activation='sigmoid')(p)

    # threshold map
    t = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', use_bias=False)(fuse)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), kernel_initializer='he_normal', use_bias=False)(t)
    t = layers.BatchNormalization()(t)
    t = layers.ReLU()(t)
    t = layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), kernel_initializer='he_normal',
                               activation='sigmoid')(t)

    # approximate binary map
    b_hat = layers.Lambda(lambda x: 1 / (1 + tf.exp(-k * (x[0] - x[1]))))([p, t])

    loss = layers.Lambda(db_loss, name='db_loss')([p, b_hat, gt_input, mask_input, t, thresh_input, thresh_mask_input])
    training_model = models.Model(inputs=[image_input, gt_input, mask_input, thresh_input, thresh_mask_input],
                                  outputs=loss)
    prediction_model = models.Model(inputs=image_input, outputs=p)
    return training_model, prediction_model



def resize_image(image, image_short_side=736):
    """
    Resize the image while maintaining aspect ratio.

    Args:
       image (numpy.ndarray): Original image.
       image_short_side (int): Size of the shorter side after resizing.

    Returns:
       numpy.ndarray: Resized image.
    """
    height, width, _ = image.shape
    if height < width:
        new_height = image_short_side
        new_width = int(math.ceil(new_height / height * width / 32) * 32)
    else:
        new_width = image_short_side
        new_height = int(math.ceil(new_width / width * height / 32) * 32)
    resized_img = cv2.resize(image, (new_width, new_height))
    return resized_img


def box_score_fast(bitmap, _box):
    # 计算 box 包围的区域的平均得分
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = np.array(offset.Execute(distance))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
           points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def polygons_from_bitmap(pred, bitmap, dest_width, dest_height, max_candidates=100, box_thresh=0.7):
    pred = pred[..., 0]
    bitmap = bitmap[..., 0]
    height, width = bitmap.shape
    boxes = []
    scores = []

    _, contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours[:max_candidates]:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape((-1, 2))
        if points.shape[0] < 4:
            continue
        score = box_score_fast(pred, points.reshape(-1, 2))
        if box_thresh > score:
            continue
        if points.shape[0] > 2:
            box = unclip(points, unclip_ratio=2.0)
            if len(box) > 1:
                continue
        else:
            continue
        box = box.reshape(-1, 2)
        _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
        if sside < 5:
            continue

        box[:, 0] = np.clip(np.round(box[:, 0] / width * dest_width), 0, dest_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / height * dest_height), 0, dest_height)
        boxes.append(box.tolist())
        scores.append(score)
    return boxes, scores


if __name__ == '__main__':
    # Parse command line arguments for input and output directories.
    args = parse_args()

    mean = np.array([103.939, 116.779, 123.68])

    _, model = dbnet()
    model.load_weights('./model.h5', by_name=True, skip_mismatch=True)
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    for image_path in glob.glob(osp.join(input_dir, '*.png')) + glob.glob(osp.join(input_dir, '*.jpg')):
        image = cv2.imread(image_path)
        src_image = image.copy()
        h, w = image.shape[:2]
        image = resize_image(image)
        image = image.astype(np.float32)
        image -= mean
        image_input = np.expand_dims(image, axis=0)
        p = model.predict(image_input)[0]
        bitmap = p > 0.3
        boxes, scores = polygons_from_bitmap(p, bitmap, w, h, box_thresh=0.5)
        
        # Create an empty mask image
        mask = np.zeros(src_image.shape[:2], dtype=np.uint8)

        # Draw the contours of the green boxes on the mask
        for box in boxes:
            cv2.drawContours(mask, [np.array(box, dtype=np.int32)], -1, 255, thickness=cv2.FILLED)
        
        # Set pixels outside the contours to zero
        src_image[np.where(mask == 0)] = 0
        
        # Save the modified image with zeros outside the contours
        image_fname = osp.split(image_path)[-1]
        output_path = osp.join(output_dir, image_fname)
        cv2.imwrite(output_path, src_image)


