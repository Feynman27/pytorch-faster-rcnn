#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

import torch

#CLASSES = ('__background__',
#           'aeroplane', 'bicycle', 'bird', 'boat',
#           'bottle', 'bus', 'car', 'cat', 'chair',
#           'cow', 'diningtable', 'dog', 'horse',
#           'motorbike', 'person', 'pottedplant',
#           'sheep', 'sofa', 'train', 'tvmonitor')
CLASSES = ('__background__', 'suiting_blazers', 'hoodies',
                 'shoes', 'messengerbags', 'jeans', 'tanks_camis', 'tunics',
                 'coats_jackets', 'cowboyhats', 'handbags', 'scarves_wraps',
                 'sweater', 'dresses', 'pants', 'clutches', 'shorts',
                 'leggings', 'boots', 'jumpsuits_rompers_overalls', 'sandals',
                 'tees', 'totes', 'belts', 'beanieknitcaps', 'bucket',
                 'slippers', 'blouses_shirts', 'skirts', 'glasses')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_%d.pth',),'res101': ('res101_faster_rcnn_iter_%d.pth',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',), 'markable_detect': ('markable_detect_trainval')}
fig,ax=None,None
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        print(class_name)
        score = dets[i, -1]
        print(bbox, score)

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    #timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(torch.from_numpy(dets), NMS_THRESH)
        dets = dets[keep.numpy(), :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time(), boxes.shape[0]))

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712 markable_detect]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    dataset = args.dataset
#    saved_model = os.path.join('output_res101_markable_v02_20171005', demonet, DATASETS[dataset][0], 'default',
#                              NETS[demonet][0] %(70000 if dataset == 'pascal_voc' else 110000))
#    saved_model = "/mnt/disk2/model_store/output_res101_markable_v02_20171005/res101/markable_detect_trainval/default/res101_faster_rcnn_iter_300000.pth"
    saved_model = "/mnt/disk2/model_store/output_20171106113426_res101_fasterrcnn_dilation/res101/markable_detect_trainval/20171106113426/res101_faster_rcnn_iter_300000.pth"
    #saved_model = "output_res101_markable_v01_20170925/res101/markable_detect_trainval/default/res101_faster_rcnn_iter_300000.pth"


    if not os.path.isfile(saved_model):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(saved_model))

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture(len(CLASSES),
                          tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(torch.load(saved_model))

    net.eval()
    net.cuda()

    print('Loaded network {:s}'.format(saved_model))

    #im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #            '001763.jpg', '004545.jpg']
    im_names = ['fall-2105-trends-sleeveess-coat-song-of-style-h724.jpg',
                'fall-2105-trends-ankle-chelsea-boots-we-wore-what-h724.jpg',
                'fall-2105-trends-burgundy-color-trend-getty-images-h724.jpg',
                'fall-2105-trends-fringe-bag-getty-images-h724.jpg',
                'fall-2105-trends-plaid-shirt-luella-june-h724.jpg',
                'fall-2105-trends-sleeveless-turtleneck-my-bubba-and-me-h724.jpg',
                'fall-2105-trends-suede-skirt-fashion-me-now-h724.jpg']
    im_names = ['26943895-12ac-4eab-8398-66b7e8addea0.png',
                '6769073-1-pink.jpeg',
                '2017DEC1_m_landing_img3.jpeg',
                '2017DEC1_m_landing_img1.jpeg']
    im_names = ['demi-high-cold-shoulder-cocktail-dress_teal_1.jpg']
    for im_name in im_names:
        fig, ax = plt.subplots(figsize=(12, 12))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(net, im_name)

    plt.show()
