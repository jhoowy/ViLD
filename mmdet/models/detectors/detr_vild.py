import warnings

import torch

from ..builder import DETECTORS
from .base import BaseDetector
from .detr import DETR


@DETECTORS.register_module()
class DETRViLD(DETR):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    # over-write `forward_train` because:
    # the forward of bbox_head requires gt_embeds
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_embeds,
                      gt_embed_weights=None,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_embeds (list[Tensor]): Image embeddings corresponding to each box
            gt_embed_weights (list[Tensor]): Weight of image embeddings for weighted
                embedding loss.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        BaseDetector.forward_train(self, img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_embeds,
                                              gt_embed_weights,
                                              gt_bboxes_ignore)
        return losses

@DETECTORS.register_module()
class DeformableDETRViLD(DETRViLD):

    def __init__(self, *args, **kwargs):
        super(DETR, self).__init__(*args, **kwargs)