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

初步测试了caffe框架和tensorrt框架运行速度。
其中caffe框架下速度特别慢
，速度慢的原因有人说是caffe对mobilenet支持不好，
并且使用caffe加载模型的时候，虽然结果是正确的，但是显存占了8G左右。
原因还没找到，有可能是模型转换出现问题，也有可能caffe的bug。
采用tensorrt进行测试，速度比mxnet快
(注：设置最大批量数为１的时候速度28ms，设置最大批量数为２，速度为58ms，批量数设置越大，速度越慢)。
具体结果见下表：

|   模型   |  速度   |   输入大小   |  预处理时间  | 推理速度 | 后处理时间
| :------: | :-----: | :---:       | :---: | :---:  | :---:
|  mxnet   | 44.8ms  |  1280ｘ896   |   19.0ms  |  8.0ms    | 16.0ms  |
|  caffe   | 46.9ms  |  1280ｘ896   |   5.8ms   |  24.1ms   | 16.0ms  |
| tensorrt | 29.3ms  |  1280ｘ896   |   6.9ms   |  5.4ms    | 15.0ms  |

|   模型   |  速度   |   输入大小   |  预处理时间  | 推理速度 | 后处理时间
| :------: | :-----: | :---:       | :---: | :---:  | :---:
|  mxnet   | 6.4ms   |  320x416   |   1.3ms   |  0.1ms    |  4.2ms  |
|  caffe   | 30.8ms  |  320x416   |   1.2ms   |  27ms     |  2.3ms  |
| tensorrt | 4.7ms  |  320x416    |   0.7ms   |  1.9ms    |  1.8ms  |

循环测试1000次。大佬说caffe对mobilenet支持不好。


trt批量处理测试：
|   batchsize  |   输入大小   |  maxbatchsize    | 预处理速度 |　推理速度 | 后处理速度 | 总时间 | GPU利用率
| :------: | :-----: | :---: | :----: | :----:  | :---: | :---: | :---: |
|  1   |   448x448   |   8   |  1.0ms  |  2.3ms  | 2.6ms | 6.7ms     | 35%
|  2   |   448x448   |   8   |  2.5ms  |  3.3ms  | 5.2ms | 11.8ms    | 33%
|  4   |   448x448   |   8   |  4.1ms  |  4.6ms  | 10.0ms| 21.8ms    | 28%
|  8   |   448x448   |   8   |  8.7ms  |  7.0ms  | 20.3ms| 40.7ms    | 23%
|  16  |   448x448   |   32   |  28.1   |  14.7   | 38.7ms| 92.0ms   | -
|  32  |   448x448   |   32   |  36.2ms |  26.3   | 75.7ms| 163.5ms  | -
注：推理时间在批量上有优势，但是预处理和后处理也要占用比较长时间。

## 精度

