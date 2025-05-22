## Records
### traning gold_yolo-n on coco
#### n8:
    * command:
        ``` bash
        torchrun --nproc_per_node=1 tools/train.py --img-size 320 --batch 64 --workers 2 --conf configs/gold_yolo-n.py --data data/coco.yaml --epoch 300 --fuse_ab --use_syncbn --device 0 --name gold_yolo-n
        ```
    * 200:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.227
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.344
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.242
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.057
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.236
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.384
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.236
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.391
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.140
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.684
        Final mAP@0.5: 0.344, mAP@0.5:0.95: 0.227
        Final mAP@0.5: 0.344, mAP@0.5:0.95: 0.227
        Epoch: 200 | mAP@0.5: 0.344050595596906 | mAP@0.50:0.95: 0.22735681583671746
        ```
#### n11: Training completed in 59.167 hours.
    * command:
        ``` bash
        torchrun --nproc_per_node=1 tools/train.py --img-size 320 --batch 64 --workers 2 --conf configs/gold_yolo-n.py --data data/coco.yaml --epoch 300 --fuse_ab --use_syncbn --device 0 --name gold_yolo-n
        ```
    * 297:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.273
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.405
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.289
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.068
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.293
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.460
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.258
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.458
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.154
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.530
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.733
        Final mAP@0.5: 0.405, mAP@0.5:0.95: 0.273
        Final mAP@0.5: 0.405, mAP@0.5:0.95: 0.273
        Epoch: 297 | mAP@0.5: 0.40522101640306163 | mAP@0.50:0.95: 0.27337406819451043
        ```
#### n13: Training completed in 167.134 hours.
    * command:
        ``` bash
        torchrun --nproc_per_node=1 tools/train.py --img-size 640 --batch 16 --workers 2 --conf configs/gold_yolo-n.py --data data/coco.yaml --epoch 300 --fuse_ab --use_syncbn --device 0 --name gold_yolo-n
        ```
    * 297:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.548
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.423
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.190
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.437
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.565
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.328
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.597
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.669
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.799
        Final mAP@0.5: 0.548, mAP@0.5:0.95: 0.391
        Final mAP@0.5: 0.548, mAP@0.5:0.95: 0.391
        Epoch: 297 | mAP@0.5: 0.5477973052286111 | mAP@0.50:0.95: 0.391483615858324
        ```
#### n22 Self-distillation training:
    * command:
        ```
        torchrun --nproc_per_node 1 tools/train.py --batch 32 --conf configs/gold_yolo-n.py --data data/coco.yaml --epoch 300 --device 0 --use_syncbn --distill --teacher_model_path runs/train/gold_yolo-n13/weights/best_ckpt.pt --name gold_yolo-n
        ```
    * 297:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.554
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.426
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.191
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.440
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.573
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.329
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.545
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.595
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.357
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.666
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.802
        Final mAP@0.5: 0.554, mAP@0.5:0.95: 0.394
        Final mAP@0.5: 0.554, mAP@0.5:0.95: 0.394
        Epoch: 297 | mAP@0.5: 0.554071975689958 | mAP@0.50:0.95: 0.3943599204609111
        ```

### traning gold_yolo-s on coco
#### s2: Training completed in 181.786 hours.
    * command:
        ```
        torchrun --nproc_per_node=1 tools/train.py --img-size 640 --batch 16 --workers 2 --conf configs/gold_yolo-s.py --data data/coco.yaml --epoch 300 --fuse_ab --use_syncbn --device 0 --name gold_yolo-s
        ```

    * 297:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.442
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.612
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.477
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.234
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.484
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.621
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.352
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.584
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.635
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.438
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.702
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.804
        Final mAP@0.5: 0.612, mAP@0.5:0.95: 0.442
        Final mAP@0.5: 0.612, mAP@0.5:0.95: 0.442
        Epoch: 297 | mAP@0.5: 0.6115975308492336 | mAP@0.50:0.95: 0.4419003360681076
        ```

#### s3: Self-distillation training:
    * command:
        ```
        torchrun --nproc_per_node 1 tools/train.py --batch 16 --conf configs/gold_yolo-s.py --data data/coco.yaml --epoch 300 --device 0 --use_syncbn --distill --teacher_model_path runs/train/gold_yolo-s2/weights/best_ckpt.pt --name gold_yolo-s
        ```
    * 297:(done)