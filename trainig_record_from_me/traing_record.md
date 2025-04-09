## Records
### traning gold_yolo-n on coco
* command:
    ``` bash
    torchrun --nproc_per_node=1 tools/train.py --img-size 320 --batch 64 --workers 2 --conf configs/gold_yolo-n.py --data data/coco.yaml --epoch 300 --fuse_ab --use_syncbn --device 0 --name gold_yolo-n
    ```
    * 20 epoch result:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.119
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.188
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.124
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.031
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.117
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.193
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.179
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.318
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.343
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.086
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.382
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.567
        Final mAP@0.5: 0.188, mAP@0.5:0.95: 0.119
        Final mAP@0.5: 0.188, mAP@0.5:0.95: 0.119
        Epoch: 20 | mAP@0.5: 0.18794027736395844 | mAP@0.50:0.95: 0.11926810046673145
        ```