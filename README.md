# ViLD
Zero-Shot Detection via Vision and Language Knowledge Distillation

## Requirements
* python >= 3.7
* torch == 1.8.1
* CUDA == 10.1
* Other requirements
    ```
    pip install -r requirements.txt
    ```

If following error occurs,
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

try install libgl1-mesa-glx by following command.
```
apt-get update
apt-get install libgl1-mesa-glx
```

## Dataset preparation

Please download [LVIS dataset](https://www.lvisdataset.org/) and organize them as following:

```
data_root/
├── train2017/
├── val2017/
└── annotations/
    ├── lvis_v1_train.json
    └── lvis_v1_val.json
```

## Preparing embeddings

Before start to training, image and text embeddings must be prepared.

1. Preparing image embeddings:
    ```
    python CLIP_embedder/img2emb.py --data_root ${DATA_ROOT}
    ```

2. Preparing text embeddings:
    ```
    python CLIP_embedder/text2emb.py --data_root ${DATA_ROOT}
    ```

3. Preparing embedding weights:
    ```
    python CLIP_embedder/calc_emb_score.py --data_root ${DATA_ROOT}
    ```

## Training

### Training on single node

```
PORT=$PORT ./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPUS}
```

For example, the command for training Mask-RCNN-ViLD on 4 GPUs is as following:

```
PORT=$PORT ./tools/dist_train.sh configs/vild/mask_rcnn_r50_vild_sl_lvis.py 4
```

### Training on multiple node


```
PORT=$PORT ./tools/dist_train_multi.sh \
    ${CONFIG_FILE} \
    ${GPUS} \
    ${NNODES} \
    ${NODE_RANK} \
    ${ADDRESS}
```

For example, the command for training Mask-RCNN-ViLD on 2 nodes of each with 4 GPU is as following:

On node 1:
```
PORT=${PORT} ./tools/dist_train_multi.sh configs/vild/mask_rcnn_r50_vild_sl_lvis.py \
    4 2 0 ${ADDRESS}
```

On node 2:
```
PORT=${PORT} ./tools/dist_train_multi.sh configs/vild/mask_rcnn_r50_vild_sl_lvis.py \
    4 2 1 ${ADDRESS}
```

## Test

### Testing on single GPU

```
PORT=$PORT python tools/test.py \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}] \
    [--show-dir ${SHOW_DIR}]
```

* `--show-dir`: If speicfied, detection results will be plotted on the images and saved to the specified directory. (Applicable to single GPU testing only)

### Testing on multiple GPU

```
PORT=$PORT ./tools/dist_test.sh \
    ${CONFIG_FILE} \
    ${CHECKPOINT_FILE} \
    ${GPUS}
    [--out ${RESULT_FILE}] \
    [--eval ${EVAL_METRICS}]
```


$EVAL_METRICS should be `bbox` or `segm`


## NOTE

There is a bug in PyTorch==1.8.1

You should modify `torch/optim/adamw.py` to train DETR-ViLD model with adamw.

Specifically, the following line 100 should move to line 76.

`beta1, beta2 = group['betas']`