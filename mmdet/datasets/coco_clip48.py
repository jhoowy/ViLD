import os
import os.path as osp

import numpy as np
from mmdet.datasets.builder import DATASETS

from mmdet.datasets import CocoDataset

@DATASETS.register_module()
class CocoCLIP48Dataset(CocoDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat', 
               'bench', 'bird', 'horse', 'sheep', 'zebra', 'giraffe', 'backpack',
               'handbag', 'skis', 'kite', 'surfboard', 'bottle', 'spoon', 'bowl', 
               'banana', 'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 
               'chair', 'bed', 'tv', 'laptop', 'remote', 'microwave', 'oven', 
               'refrigerator', 'book', 'clock', 'vase', 'toothbrush', 'train', 
               'bear', 'suitcase', 'frisbee', 'fork', 'sandwich', 'toilet', 'mouse',
               'toaster')

    def __init__(self, emb_prefix=None, data_root=None, **kwargs):
        self.emb_prefix = emb_prefix

        # join paths if data_root is specified
        if data_root is not None:
            if not (self.emb_prefix is None or osp.isabs(self.emb_prefix)):
                self.emb_prefix = osp.join(data_root, self.emb_prefix)

        super(CocoCLIP48Dataset, self).__init__(data_root=data_root, **kwargs)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.
        Image embedding is added in this dataset.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        emb_filename = '.'.join((img_info['filename'].split('.')[0], 'pickle'))
        results = dict(img_info=img_info, ann_info=ann_info, 
                       emb_prefix=self.emb_prefix, emb_filename=emb_filename)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation. Annotation id is added to load
        iamge embedding later.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map, ann_ids. "masks" are raw annotations \
                and not decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_ann_ids = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))
                gt_ann_ids.append(ann['id'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_ann_ids = np.array(gt_ann_ids, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_ann_ids = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map,
            ann_ids=gt_ann_ids)

        return ann