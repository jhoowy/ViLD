## ViLD
Zero-Shot Detection via Vision and Language Knowledge Distillation

## Training

`./tools/dist_train.sh configs/vild/faster_rcnn_r50_vild_lvis.py $GPUS`



# NOTE

There is a bug in PyTorch==1.8.1

You should modify `torch/optim/adamw.py` to train DETR-ViLD model with adamw.

Specifically, the following line 100 should move to line 76.

`beta1, beta2 = group['betas']`