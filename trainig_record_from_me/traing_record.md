## Records
### traning gold_yolo-n on coco
* command:
    ``` bash
    torchrun --nproc_per_node=1 tools/train.py --img-size 320 --batch 64 --workers 2 --conf configs/gold_yolo-n.py --data data/coco.yaml --epoch 300 --fuse_ab --use_syncbn --device 0 --name gold_yolo-n
    ```
    #### n8:
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
        * 40 epoch result:
            ```
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.152
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.239
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.159
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.040
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.155
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.250
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.198
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.348
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.378
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.105
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.430
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.617
            Final mAP@0.5: 0.239, mAP@0.5:0.95: 0.152
            Final mAP@0.5: 0.239, mAP@0.5:0.95: 0.152
            Epoch: 40 | mAP@0.5: 0.23919950181521335 | mAP@0.50:0.95: 0.1521802896272059
            ```
        * 60 epoch result:
            ```
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.167
            Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.261
            Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.176
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
            Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.170
            Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.275
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.205
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.357
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.389
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.111
            Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.443
            Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.631
            Final mAP@0.5: 0.261, mAP@0.5:0.95: 0.167
            Final mAP@0.5: 0.261, mAP@0.5:0.95: 0.167
            Epoch: 60 | mAP@0.5: 0.2612773929912459 | mAP@0.50:0.95: 0.1674588172033737
            ```
        * 80 epoch result:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.177
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.275
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.186
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.182
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.293
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.210
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.361
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.116
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.448
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.642
        Final mAP@0.5: 0.275, mAP@0.5:0.95: 0.177
        Final mAP@0.5: 0.275, mAP@0.5:0.95: 0.177
        Epoch: 80 | mAP@0.5: 0.27532124029562244 | mAP@0.50:0.95: 0.17689302881460972
        ```
        * 120:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.192
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.296
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.203
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.047
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.323
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.217
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.370
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.404
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.121
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.462
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651
        Final mAP@0.5: 0.296, mAP@0.5:0.95: 0.192
        Final mAP@0.5: 0.296, mAP@0.5:0.95: 0.192
        Epoch: 120 | mAP@0.5: 0.296216257284391 | mAP@0.50:0.95: 0.19190249285940328
        ```
        * 140:
        ```
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.201
        Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.309
        Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.211
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.049
        Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.206
        Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.340
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.223
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.375
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.410
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.124
        Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.467
        Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.660
        Final mAP@0.5: 0.309, mAP@0.5:0.95: 0.201
        Final mAP@0.5: 0.309, mAP@0.5:0.95: 0.201
        Epoch: 140 | mAP@0.5: 0.3094196664085125 | mAP@0.50:0.95: 0.2008188574325598
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