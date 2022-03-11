import sys
if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle5 as pickle
from mmdet.models.builder import HEADS

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.roi_heads import StandardRoIHead


@HEADS.register_module()
class CLIPRoIHead(StandardRoIHead):
    """ViLD: Vision and Language Knowledge Distillation
    RoI head for CLIP on cropped region.

    https://arxiv.org/abs/2104.13921
    """

    def __init__(self, novel_emb_path, with_ens=False, *args, **kwargs):
        super(CLIPRoIHead, self).__init__(*args, **kwargs)
        self.bg_embedding = nn.Parameter(torch.zeros((1, 512)))
        self._load_text_embedding(novel_emb_path)
        self.with_ens = with_ens

    def _load_text_embedding(self, emb_path):
        with open(emb_path, 'rb') as f:
            text_embeddings = pickle.load(f)

        text_embeddings = torch.from_numpy(text_embeddings).float()
        if hasattr(self, 'text_embeddings'):
            text_embeddings = text_embeddings.to(self.text_embeddings.device)
        text_embeddings.requires_grad = False
        self.register_buffer('text_embeddings', text_embeddings, persistent=False)
        return self.text_embeddings

    def simple_test(self,
                    x,
                    proposal_list,
                    gt_embeds,
                    gt_embed_scores,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, gt_embeds, gt_embed_scores, proposal_list, 
            self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    async def async_simple_test(self,
                                x,
                                proposal_list,
                                gt_embeds,
                                gt_embed_scores,
                                img_metas,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = await self.async_test_bboxes(
            x, img_metas, gt_embeds, gt_embed_scores, proposal_list, 
            self.test_cfg, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = await self.async_test_mask(
                x,
                img_metas,
                det_bboxes,
                det_labels,
                rescale=rescale,
                mask_test_cfg=self.test_cfg.get('mask'))
            return bbox_results, segm_results

    async def async_test_bboxes(self,
                                x,
                                img_metas,
                                gt_embeds,
                                gt_embed_scores,
                                proposals,
                                rcnn_test_cfg,
                                rescale=False,
                                **kwargs):
        """Asynchronized test for box head without augmentation."""
        
        for i in range(len(proposals)):
            proposals[i] = proposals[i][0]
            gt_embeds[i] = gt_embeds[i][0]
            if self.with_ens:
                gt_embed_scores[i] = gt_embed_scores[i][0]
            else:
                gt_embed_scores[i] = None
        rois = bbox2roi(proposals)
        roi_feats = self.bbox_roi_extractor(
            x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
        if self.with_shared_head:
            roi_feats = self.shared_head(roi_feats)
        sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

        async with completed(
                __name__, 'bbox_head_forward',
                sleep_interval=sleep_interval):
            cls_score, bbox_pred = self.bbox_head(roi_feats)
            
        cls_emb = torch.cat((self.text_embeddings, self.bg_embedding), dim=0)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        cls_score = 100. * gt_embeds @ cls_emb.t()

        img_shape = img_metas[0]['img_shape']
        scale_factor = img_metas[0]['scale_factor']
        det_bboxes, det_labels = self.bbox_head.get_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            obj_score=gt_embed_scores[i],
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
    
    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           gt_embeds,
                           gt_embed_scores,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        for i in range(len(proposals)):
            proposals[i] = proposals[i][0]
            gt_embeds[i] = gt_embeds[i][0]
            if self.with_ens:
                gt_embed_scores[i] = gt_embed_scores[i][0]
            else:
                gt_embed_scores[i] = None
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)
            
        gt_embeds = torch.cat(gt_embeds, dim=0)
        cls_emb = torch.cat((self.text_embeddings, self.bg_embedding), dim=0)
        cls_score = 100. * gt_embeds @ cls_emb.t()

        # split batch bbox prediction back to each image
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    obj_score=gt_embed_scores[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels