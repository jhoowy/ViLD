import os
import os.path as osp
import numpy as np
import pickle5 as pickle
from tqdm import tqdm

from lvis import LVIS

import torch
import clip
from PIL import Image
import argparse

from mmdet.apis import init_detector, inference_detector

class ImageCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(ImageCLIP, self).__init__()
        self.model = model
        
    def forward(self,image):
        return self.model.encode_image(image)

parser = argparse.ArgumentParser(description='Create image embedding for ViLD')
parser.add_argument('--data_root', default='/data/project/rw/lvis_v1', type=str)
parser.add_argument('--proposal_dir', default='proposals', type=str)
parser.add_argument('--save_dir', default='proposal_embeddings', type=str)
args = parser.parse_args()

data_root = args.data_root
ann_file = osp.join(data_root, 'annotations/lvis_v1_train.json')
save_dir = osp.join(data_root, args.save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model_image = torch.nn.DataParallel(ImageCLIP(model))

os.makedirs(osp.join(save_dir, 'train2017'), exist_ok=True)
os.makedirs(osp.join(save_dir, 'val2017'), exist_ok=True)

coco = LVIS(ann_file)
img_ids = coco.get_img_ids()
img_ids = img_ids

for i in tqdm(img_ids):
    img_info = coco.load_imgs([i])[0]
    filename = img_info['coco_url'].replace('http://images.cocodataset.org/', '')

    img = Image.open(osp.join(data_root, filename))
    outname = '.'.join((filename.split('.')[0], 'pickle'))

    p_filename = '.'.join((filename.split('.')[0] + '_p', 'pickle'))
    proposal_path = osp.join(data_root, args.proposal_dir, p_filename)
    with open(proposal_path, 'rb') as f:
        proposals = pickle.load(f)

    results = {}
    embeddings = []
    embed_bboxes = []
    embed_scores = []
    batch = []
    batch_l = []
    for p in proposals:
        x1, y1, x2, y2, s = p.numpy()
        w = x2 - x1
        h = y2 - y1
        
        if w < 1 or h < 1:
            continue

        # x1.5 crop
        lx1 = max(0, x1 - w//2)
        ly1 = max(0, y1 - h//2)
        lx2 = min(x1 + w + w//2, img_info['width'])
        ly2 = min(y1 + h + h//2, img_info['height'])

        im_crop = img.crop((x1, y1, x2, y2))
        im_crop = preprocess(im_crop).unsqueeze(0).to(device)
        im_crop_l = img.crop((lx1, ly1, lx2, ly2))
        im_crop_l = preprocess(im_crop_l).unsqueeze(0).to(device)

        batch.append(im_crop)
        batch_l.append(im_crop_l)
        embed_bboxes.append(p[:4])
        embed_scores.append(p[4])

    with torch.no_grad():
        batch_size = len(batch)
        input = torch.cat(batch + batch_l)
        im_embed = model_image(input)
        im_embed = im_embed[:batch_size] + im_embed[batch_size:]
        im_embed /= im_embed.norm(dim=-1, keepdim=True)
    embeddings.append(im_embed)
    batch = []
    batch_l = []
    
    embeddings = torch.cat(embeddings).cpu().numpy()
    embed_bboxes = torch.stack(embed_bboxes).cpu().numpy()
    embed_scores = torch.stack(embed_scores).cpu().numpy()

    results['img_embeds'] = embeddings
    results['bboxes'] = embed_bboxes
    results['scores'] = embed_scores
    
    with open(osp.join(save_dir, outname), 'wb') as out:
        pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)