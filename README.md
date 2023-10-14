## What is this
TopDown Pose Estimation on iOS and PureSwift.  

- BBox: Yolov7-tiny
- Pose Estimation: ViTPose

## Install

```
$ git clone https://github.com/otmb/TopDownPoseEstimation.git
$ cd TopDownPoseEstimation/TopDownPoseEstimation
$ curl -OL https://github.com/mbotsu/KeypointDecoder/releases/download/0.0.1/vitpose-b256x192_fp16.mlmodel
$ curl -OL https://github.com/mbotsu/KeypointDecoder/releases/download/0.0.1/yolov7-tiny_fp16.mlmodel
```

## Example

<img src="sample.png" width="500" />

## COCO MS val set

Unrated

## Sample Models

| Models | Size | Keypoint | 
|--|--|:--:|
| [vitpose-b256x192_fp16.mlmodel](https://github.com/mbotsu/KeypointDecoder/releases/download/0.0.1/vitpose-b256x192_fp16.mlmodel) | 172MB | 17 |
| [vitpose_s256x192_wholebody_fp16.mlmodel](https://github.com/mbotsu/KeypointDecoder/releases/download/0.0.6/vitpose_s256x192_wholebody_fp16.mlmodel)| 46.5MB | 133 |
| [vitpose_b256x192_wholebody_fp16.mlmodel](https://github.com/mbotsu/KeypointDecoder/releases/download/0.0.1/vitpose_b256x192_wholebody_fp16.mlmodel) | 172MB | 133 |
| [yolov7-tiny_fp16.mlmodel](https://github.com/mbotsu/KeypointDecoder/releases/download/0.0.1/yolov7-tiny_fp16.mlmodel) | 12.1MB | - |

[COCO-Wholebody](https://github.com/jin-s13/COCO-WholeBody) 133 When using Keypoint, change the following after introducing the model to the project.

Edit: PoseEstimation.swift 
- keypointsNumber
- modelName

## Create Model References

- ViTPose to CoreML
    - [mbotsu/20221128_convert.ipynb](https://gist.github.com/mbotsu/3de024c36582f21306e23473e9975841)
- Yolov7 to CoreML
    - [john-rocky/CoreML-Models](https://github.com/john-rocky/CoreML-Models#yolov7) 
        - [Yolov7 Google Colab](https://colab.research.google.com/drive/1QiTlFsN948Xt2e4WgqUB8DnGgwWwtVZS?usp=sharing)

## References
- [microsoft/human-pose-estimation.pytorch](https://github.com/microsoft/human-pose-estimation.pytorch)
- [PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/blob/develop/deploy/lite/src/keypoint_postprocess.cc)
- [ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- ViTPose to CoreML
    - [mbotsu/20221128_convert.ipynb](https://gist.github.com/mbotsu/3de024c36582f21306e23473e9975841)
- [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- Yolov7 to CoreML
    - [john-rocky/CoreML-Models](https://github.com/john-rocky/CoreML-Models#yolov7)
- AffineTransform
    - [Perspective transform from quadrilateral to quadrilateral in Swift](https://rethunk.medium.com/perspective-transform-from-quadrilateral-to-quadrilateral-in-swift-5a9adf2175c3)
- Drawing processing
    - [Detecting human body poses in an image](https://developer.apple.com/documentation/coreml/model_integration_samples/detecting_human_body_poses_in_an_image)
