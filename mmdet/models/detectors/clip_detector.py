import warnings

import torch

from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class CLIPDetector(TwoStageDetector):
    """ViLD: Vision and Language Knowledge Distillation

    CLIP on the cropped region, only support simple test function.
    It cannot be trained by itself.

    https://arxiv.org/abs/2104.13921
    """

    def simple_test(self, 
                    img, 
                    img_metas, 
                    gt_embeds, 
                    gt_embed_bboxes, 
                    gt_embed_scores, 
                    rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        proposal_list = gt_embed_bboxes

        return self.roi_head.simple_test(
            x, proposal_list, gt_embeds, gt_embed_scores,
            img_metas, rescale=rescale)

    async def async_simple_test(self,
                                img,
                                img_metas,
                                gt_embeds,
                                gt_embed_bboxes,
                                gt_embed_scores,
                                rescale=False):
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        proposal_list = gt_embed_bboxes

        return await self.roi_head.async_simple_test(
            x, proposal_list, gt_embeds, gt_embed_scores,
            img_metas, rescale=rescale)