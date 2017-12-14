# --------------------------------------------------------
# Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from model.config import cfg
from model.bbox_transform import bbox_transform_inv, clip_boxes
from model.nms_wrapper import nms

import torch
from torch.autograd import Variable


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, cfg_key, _feat_stride, anchors, num_anchors):
    """A simplified version compared to fast/er RCNN
     For details please see the technical report
    """
    if type(cfg_key) == bytes:
        cfg_key = cfg_key.decode('utf-8')
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH

    im_info = im_info[0]
    # Get the foreground scores and bounding boxes
    scores = rpn_cls_prob[:, :, :, num_anchors:]
    # Reshape bbox preds to [A*H*W, 4]
    rpn_bbox_pred = rpn_bbox_pred.view((-1, 4))
    # Reshape fg scores to [A*H*W, 1]
    scores = scores.contiguous().view(-1, 1)
    # Solve for x,y,w,h of predicted bboxes [A*H*W, 4]
    proposals = bbox_transform_inv(anchors, rpn_bbox_pred)
    # Clip predicted boxes to original image size
    proposals = clip_boxes(proposals, im_info[:2])

    # Pick the top region proposals
    scores, order = scores.view(-1).sort(descending=True)
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
        scores = scores[:pre_nms_topN].view(-1, 1)
    # Top pre_nms_topN proposals
    proposals = proposals[order.data, :]

    # Non-maximal suppression
    keep = nms(torch.cat((proposals, scores), 1).data, nms_thresh)

    # Pick the top region proposals after NMS
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep,]

    # Only support single image as input
    batch_inds = Variable(proposals.data.new(proposals.size(0), 1).zero_())
    blob = torch.cat((batch_inds, proposals), 1)
    # Return blob of size [post_nms_topN, 5] and scores [post_nms_topN,1]
    return blob, scores
