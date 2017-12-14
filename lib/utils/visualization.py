# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from model.bbox_transform import clip_boxes, bbox_transform_inv
from model.nms_wrapper import nms

import torch

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]


def draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font, color='black', thickness=4):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)
    draw.rectangle(
        [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                          text_bottom)],
        fill=color)
    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill='black',
        font=font)

    return image


def _clip_boxes(boxes, im_shape):
    """Clip boxes to image boundaries."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(boxes[:, 0::4], 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(boxes[:, 1::4], 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.minimum(boxes[:, 2::4], im_shape[1] - 1)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.minimum(boxes[:, 3::4], im_shape[0] - 1)
    return boxes

def draw_pred_bounding_boxes(image, scores, bbox_pred, rois, im_info, num_classes=21):
    im_info = im_info[0]
    im_scale = im_info[-1]
    #boxes = rois[:, 1:5] / im_scale
    boxes = rois[:, 1:5]
    scores = np.reshape(scores, [scores.shape[0], -1])
    bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
    # Apply bounding-box regression deltas
    box_deltas = bbox_pred
    pred_boxes = bbox_transform_inv(torch.from_numpy(boxes), torch.from_numpy(box_deltas)).numpy()
    ## h,w of original image
    #h = int(np.round(image.shape[1]/im_scale))
    #w = int(np.round(image.shape[2]/im_scale))
    h = image.shape[1]
    w = image.shape[2]
    pred_boxes = _clip_boxes(pred_boxes, (h,w))

    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    num_colors = len(STANDARD_COLORS)
    disp_image = Image.fromarray(np.uint8(image[0]))
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    #for j in range(1, num_classes):
    #    inds = np.where(scores[:, j] > thresh)[0]
    #    cls_scores = scores[inds, j]
    #    cls_boxes = boxes[inds, j * 4:(j + 1) * 4]
    #    cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
    #        .astype(np.float32, copy=False)
    #    keep = nms(torch.from_numpy(cls_dets), cfg.TEST.NMS).numpy() if cls_dets.size > 0 else []
    #    cls_dets = cls_dets[keep, :]
    #    all_boxes[j][i] = cls_dets

    for cls_ind in range(1, num_classes):
        inds = np.where(scores[:, cls_ind] > CONF_THRESH)[0]
        cls_scores = scores[inds, cls_ind]
        cls_boxes = pred_boxes[inds, 4 * cls_ind:4 * (cls_ind + 1)]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = nms(torch.from_numpy(dets), NMS_THRESH).numpy() if dets.size > 0 else []
        dets = dets[keep, :]

        #inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue
#        import pdb; pdb.set_trace()
        for i in range(len(keep)):
            bbox = dets[i, :4]
            score = dets[i, -1]
            this_class = int(cls_ind)
            disp_image = draw_single_box(disp_image,
                                         bbox[0],
                                         bbox[1],
                                         bbox[2],
                                         bbox[3],
                                         'N%d-C%d' % (i, this_class),
                                         font,
                                         color=STANDARD_COLORS[this_class % num_colors])

    image[0, :] = np.array(disp_image)
    return image


def draw_bounding_boxes(image, gt_boxes, im_info):
    num_boxes = gt_boxes.shape[0]
    num_colors = len(STANDARD_COLORS)
    im_info = im_info[0]
    disp_image = Image.fromarray(np.uint8(image[0]))

    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    for i in xrange(num_boxes):
        this_class = int(gt_boxes[i, 4])
        disp_image = draw_single_box(disp_image,
                                     gt_boxes[i, 0],
                                     gt_boxes[i, 1],
                                     gt_boxes[i, 2],
                                     gt_boxes[i, 3],
                                     'N%d-C%d' % (i, this_class),
                                     font,
                                     color=STANDARD_COLORS[this_class % num_colors])

    image[0, :] = np.array(disp_image)
    return image
