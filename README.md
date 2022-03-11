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

## Data preparation

### Dataset download
Please download [LVIS dataset](https://www.lvisdataset.org/) and organize them as following:

```
data_root/
├── train2017/
├── val2017/
└── annotations/
    ├── lvis_v1_train.json
    └── lvis_v1_val.json
```

### Preparing proposals & embeddings

Before start to training and testing ViLD, precomputed region proposals,
image embeddings, and text embeddings must be prepared.

1. **Preparing region proposals**
    
    To extract region proposals, the baseline RPN model should be prepared.
    You can train it yourself by following training command:
    ```
    ./tools/dist_train.sh configs/mask_rcnn_r50_fpn_lvis.py ${GPUS}
    ```

    After train baseline model, you extract proposals for training by following command:
    ```
    ./tools/extract_proposal.sh configs/mask_rcnn_r50_fpn_proposal.py \
        ${CHECKPOINT_FILE} \
        ${GPUS} \
        --out-dir proposals
    ```

    If you want to extract proposals for validation, you should change the 67~71 line as below
    ```
    test=dict(
        type='LVISClipCFDataset',
        ann_file=data_root + 'annotations/lvis_v1_train.json',
        img_prefix=data_root,
        pipeline=test_pipeline))
    ```

2. **Preparing image embeddings**
    ```
    python CLIP_embedder/img2emb.py --data_root ${DATA_ROOT}
    ```

3. **Preparing text embeddings**
    ```
    python CLIP_embedder/text2emb.py --data_root ${DATA_ROOT}
    ```

## Training

### Training on single node

```
./tools/dist_train.sh \
    ${CONFIG_FILE} \
    ${GPUS}
```

For example, the command for training Mask-RCNN-ViLD on 4 GPUs is as following:

```
./tools/dist_train.sh configs/vild/mask_rcnn_r50_vild_lvis.py 4
```

### Training on multiple node


```
PORT=$PORT ./tools/dist_train_multi.sh \
    ${CONFIG_FILE} \
    ${GPUS} \
    ${NNODES} \
    ${NODE_RANK} \
    ${ADDRESS} \
    --zero_shot
```

For example, the command for training Mask-RCNN-ViLD on 2 nodes of each with 4 GPU is as following:

On node 1:
```
PORT=${PORT} ./tools/dist_train_multi.sh configs/vild/mask_rcnn_r50_vild_lvis.py \
    4 2 0 ${ADDRESS} --zero_shot
```

On node 2:
```
PORT=${PORT} ./tools/dist_train_multi.sh configs/vild/mask_rcnn_r50_vild_lvis.py \
    4 2 1 ${ADDRESS} --zero_shot
```

## Test

### Testing on single GPU

```
python tools/test.py \
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

### Testing CLIP on cropped regions

```
./tools/dist_test.sh configs/mask_rcnn/mask_rcnn_r50_fpn_lvis_clip.py \
    ${CHECKPOINT_FILE} \
    ${GPUS} \
    --eval segm
```