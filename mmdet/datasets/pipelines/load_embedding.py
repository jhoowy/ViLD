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

    Required keys are "emb_prefix" and "emb_filename". Added key is "gt_embeds".
    
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
                self.embed_weights[int(k)] = v

    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        if results['emb_prefix'] is not None:
            filename = osp.join(results['emb_prefix'],
                                results['emb_filename'])
        else:
            filename = results['emb_filename']

        with open(filename, 'rb') as f:
            img_embeds = pickle.load(f)

        ann_ids = results['ann_info']['ann_ids']
        gt_embeds = [img_embeds[id].astype(np.float32) for id in ann_ids]
        if len(gt_embeds) > 0:
            gt_embeds = np.concatenate(gt_embeds)
        else:
            gt_embeds = np.empty((0, 512)).astype(np.float32)
        results['gt_embeds'] = gt_embeds

        if self.with_score:
            gt_embed_weights = np.array([self.embed_weights[id] for id in ann_ids])
            results['gt_embed_weights'] = gt_embed_weights.astype(np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'file_client_args={self.file_client_args})')
        return repr_str