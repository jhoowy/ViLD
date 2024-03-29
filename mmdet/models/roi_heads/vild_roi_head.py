# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import pickle
from mmdet.models.builder import HEADS

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class ViLDRoIHead(StandardRoIHead):
    """ViLD: Vision and Language Knowledge Distillation
    TODO: Decouple with ViLDBBoxHead

    https://arxiv.org/abs/2104.13921
    """

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_embeds,
                      gt_embed_bboxes,
                      gt_embed_weights=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_embeds (list[Tensor]): Image embeddings corresponding to each box
            gt_embed_weights (list[Tensor]): Weight of image embeddings for weighted
                embedding loss.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    gt_embeds, gt_embed_bboxes,
                                                    gt_embed_weights, img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas)
            losses.update(mask_results['loss_mask'])

        return losses

    def _bbox_forward_train(self, x, sampling_results, gt_bboxes, gt_labels,
                            gt_embeds, gt_embed_bboxes, gt_embed_weights, 
                            img_metas):
        """Run forward function and calculate loss for box head in training."""
        rois = bbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        embed_rois = bbox2roi([bboxes for bboxes in gt_embed_bboxes])
        embed_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], embed_rois)
        if self.with_shared_head:
            embed_feats = self.shared_head(embed_feats)
        
        image_embeds = self.bbox_head.forward_embed(embed_feats)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets,
                                        embeds=gt_embeds,
                                        embed_weights=gt_embed_weights,
                                        region_embed=image_embeds)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results