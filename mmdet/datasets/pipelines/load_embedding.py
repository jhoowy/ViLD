import os.path as osp

import torch
import mmcv
import numpy as np
import pickle5 as pickle

from mmdet.datasets import PIPELINES

@PIPELINES.register_module()
class LoadEmbeddingFromFile:
    """Load embeddings from file.

    Required keys are "emb_prefix" and "emb_filename". Added key is "gt_embeds".
    
    Args:
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self, 
                 file_client_args=dict(backend='disk')):
        self.file_client_args = file_client_args.copy()
        self.file_client = None

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
        # gt_embeds = [torch.from_numpy(img_embeds[id]) for id in ann_ids]
        # results['gt_embeds'] = torch.cat(gt_embeds, dim=0)
        gt_embeds = [img_embeds[id] for id in ann_ids]
        results['gt_embeds'] = np.concatenate(gt_embeds)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'file_client_args={self.file_client_args})')
        return repr_str