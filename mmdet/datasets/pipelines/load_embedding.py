import os.path as osp

import torch
import json
import mmcv
import numpy as np
import pickle5 as pickle

from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class LoadEmbeddingFromFile:
    """Load embeddings from file.

    Required keys are "emb_prefix" and "emb_filename". Added key is "gt_embeds", 
    "gt_embed_bboxes", and "gt_embed_weights" (optional).
    
    Args:
        with_score (bool) : Whether to parse and load the embedding score.
            Default: False.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, 
                 with_score=False,
                 ann_file=None,
                 file_client_args=dict(backend='disk')):
        assert (ann_file is not None) or (not with_score)
        self.with_score = with_score
        self.ann_file = ann_file
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        if self.with_score:
            self.embed_weights = {}
            with open(self.ann_file, 'r') as f:
                weight_data = json.load(f)
            for k, v in weight_data.items():
                self.embed_weights[k] = v

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        if results['emb_prefix'] is not None:
            filename = osp.join(results['emb_prefix'],
                                results['emb_filename'])
        else:
            filename = results['emb_filename']

        with open(filename, 'rb') as f:
            data = pickle.load(f)

        gt_embeds = data['img_embeds']
        gt_embed_bboxes = data['bboxes']
        gt_embed_scores = data['scores']

        if len(gt_embeds) > 0:
            gt_embeds = np.concatenate(gt_embeds).astype(np.float32)
            gt_embed_bboxes = np.concatenate(gt_embed_bboxes).astype(np.float32)
            gt_embed_scores = np.concatenate(gt_embed_scores).astype(np.float32)
        else:
            gt_embeds = np.empty((0.512)).astype(np.float32)
            gt_embed_bboxes = np.empty((0, 4)).astype(np.float32)
            gt_embed_scores = np.empty((0, 1)).astype(np.float32)

        results['gt_embeds'] = gt_embeds
        results['gt_embed_bboxes'] = gt_embed_bboxes
        results['gt_embed_scores'] = gt_embed_scores
        results['bbox_fields'].append('gt_embed_bboxes')
        
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'file_client_args={self.file_client_args})')
        return repr_str