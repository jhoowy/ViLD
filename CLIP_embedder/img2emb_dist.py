import os
import os.path as osp
import numpy as np
import pickle5 as pickle
from tqdm import tqdm

import matplotlib
matplotlib.use("pdf") # Prevent hanging issue
from lvis import LVIS

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize
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

class STN(torch.nn.Module):
    """Crop images with bboxs"""
    def __init__(self, out_size, device):
        super(STN, self).__init__()
        self.out_size = out_size
        self.device = device
    
    def forward(self, bboxs, img):
        h, w = img.size(2), img.size(3)
        left = torch.clamp(bboxs[:, 0], min=0)
        bottom = torch.clamp(bboxs[:, 1], min=0)
        right = torch.clamp(bboxs[:, 2], max=w)
        top = torch.clamp(bboxs[:, 3], max=h)

        theta = torch.zeros((bboxs.size(0), 2, 3), dtype=torch.float, device=self.device) 
        
        theta[:, 0, 0] = (right-left)/w
        theta[:, 1, 1] = (top-bottom)/h
        
        theta[:, 0, 2] =  -1. +  (left + (right-left)/2)/(w/2)
        theta[:, 1, 2] =  -1. + ((bottom + (top-bottom)/2))/(h/2)
        
        grid_size = torch.Size((bboxs.size(0), img.size(1), self.out_size, self.out_size))
        grid = F.affine_grid(theta, grid_size, align_corners=False)
        objs = F.grid_sample(torch.cat(bboxs.size(0)*[img]), grid, align_corners=False)

        return objs

parser = argparse.ArgumentParser(description='Create image embedding for ViLD')
parser.add_argument('--data_root', default='/data/project/rw/lvis_v1', type=str)
parser.add_argument('--proposal_dir', default='proposals', type=str)
parser.add_argument('--save_dir', default='proposal_embeddings', type=str)
args = parser.parse_args()

data_root = args.data_root
ann_file = osp.join(data_root, 'annotations/lvis_v1_train.json')
save_dir = osp.join(data_root, args.save_dir)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _ = clip.load("ViT-B/32", device=device)
model_image = torch.nn.DataParallel(ImageCLIP(model))
stn = STN(224, device=device)

preprocess = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])

os.makedirs(osp.join(save_dir, 'train2017'), exist_ok=True)
os.makedirs(osp.join(save_dir, 'val2017'), exist_ok=True)

coco = LVIS(ann_file)
img_ids = coco.get_img_ids()

for i in tqdm(img_ids):
    img_info = coco.load_imgs([i])[0]
    filename = img_info['coco_url'].replace('http://images.cocodataset.org/', '')

    img = Image.open(osp.join(data_root, filename)).convert("RGB")
    outname = '.'.join((filename.split('.')[0], 'pickle'))

    p_filename = '.'.join((filename.split('.')[0] + '_p', 'pickle'))
    proposal_path = osp.join(data_root, args.proposal_dir, p_filename)
    with open(proposal_path, 'rb') as f:
        proposals = pickle.load(f)

    results = {}
    embed_bboxes = proposals[:, :4].numpy()
    embed_scores = proposals[:, 4:].numpy()

    bboxs = proposals[:, :4].to(device)
    l_bboxs = bboxs
    w = l_bboxs[:, 2] - l_bboxs[:, 0]
    h = l_bboxs[:, 3] - l_bboxs[:, 1]
    l_bboxs[:, 0] = l_bboxs[:, 0] - w/2
    l_bboxs[:, 1] = l_bboxs[:, 1] - h/2
    l_bboxs[:, 2] = l_bboxs[:, 2] + w/2
    l_bboxs[:, 3] = l_bboxs[:, 3] + h/2

    img = preprocess(img).to(device).unsqueeze(0)
    batch_size = bboxs.shape[0]
    with torch.no_grad():
        # TODO: Implement batch processing for STN
        imgs = stn(bboxs, img)
        l_imgs = stn(l_bboxs, img)
        input_img = torch.cat((imgs, l_imgs), dim=0)

        embed = model_image(input_img)
        embed = embed[:batch_size] + embed[batch_size:]
        embed /= embed.norm(dim=-1, keepdim=True)

    results['img_embeds'] = embed.cpu().numpy()
    results['bboxes'] = embed_bboxes
    results['scores'] = embed_scores
    
    with open(osp.join(save_dir, outname), 'wb') as out:
        pickle.dump(results, out, pickle.HIGHEST_PROTOCOL)