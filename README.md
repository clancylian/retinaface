# RetinaFace C++ 复现

## 源码
参考insightface中的[RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) python 代码

## 模型转换工具
[MXNet2Caffe](https://github.com/cypw/MXNet2Caffe)

需要自己添加一些层，caffe中没有upsample层，使用deconvition替代，会有精度损失。

原模型来源大佬提供模型：[mobilenet25](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w#list/path=%2F)，后续自己重训练。

## Demo
```bash
$ mkdir build
$ cd build/
$ cmake
$ make
```

## 速度

测试环境：1080Ti

测试图片：

|   模型   |  速度   |      |      |
| :------: | :-----: | :--: | ---- |
|  mxnet   | 39.7ms  |      |      |
|  caffe   | 811.4ms |      |      |
| tensorrt | 58.1ms  |      |      |

循环测试1000次。大佬说caffe对mobilenet支持不好。

trt批量处理：
|   batchsize   |  推理速度   |   输入大小   |      |
| :------: | :-----: | :--: | ---- |
|  1   |   |   448x448   |      |
|  2   |   |   448x448   |      |
|  4   |   |   448x448   |      |
|  8   |   |   448x448   |      |
|  16  |   |   448x448   |      |
|  32  |   |   448x448   |      |
注：推理时间在批量上有优势，但是预处理和后处理也要占用比较长时间。

## 精度

