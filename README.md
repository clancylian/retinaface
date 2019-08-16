# RetinaFace C++ Reimplement

## source
 Reference resources [RetinaFace](https://github.com/deepinsight/insightface/tree/master/RetinaFace) in insightface with python code. 

## model transformation tool
[MXNet2Caffe](https://github.com/cypw/MXNet2Caffe)

you need to add some layers yourself, and in caffe there is not **upsmaple**,you can replace with **deconvition**,and maybe slight accuracy loss.

the origin model reference from [mobilenet25](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w#list/path=%2F)，and I have retrain it.

## Demo
```
$ mkdir build
$ cd build/
$ cmake ../
$ make
```
you need to modify dependency path in CmakeList file.

## Speed

test hardware：1080Ti

test1:

|  model   | speed  | input size | preprocess time | inference | postprocess time |
| :------: | :----: | :--------: | :-------------: | :-------: | :--------------: |
|  mxnet   | 44.8ms | 1280ｘ896  |     19.0ms      |   8.0ms   |      16.0ms      |
|  caffe   | 46.9ms | 1280ｘ896  |      5.8ms      |  24.1ms   |      16.0ms      |
| tensorrt | 29.3ms | 1280ｘ896  |      6.9ms      |   5.4ms   |      15.0ms      |

test2:

|  model   | speed  | inputsize | preprocess time | inference | postprocess time |
| :------: | :----: | :-------: | :-------------: | :-------: | :--------------: |
|  mxnet   | 6.4ms  |  320x416  |      1.3ms      |   0.1ms   |      4.2ms       |
|  caffe   | 30.8ms |  320x416  |      1.2ms      |   27ms    |      2.3ms       |
| tensorrt | 4.7ms  |  320x416  |      0.7ms      |   1.9ms   |      1.8ms       |

tensorrt batch test:

| batchsize | inputsize | maxbatchsize | preprocess time | inference | postprocess time |   all   | GPU  |
| :-------: | :-------: | :----------: | :-------------: | :-------: | :--------------: | :-----: | :--: |
|     1     |  448x448  |      8       |      1.0ms      |   2.3ms   |      2.6ms       |  6.7ms  | 35%  |
|     2     |  448x448  |      8       |      2.5ms      |   3.3ms   |      5.2ms       | 11.8ms  | 33%  |
|     4     |  448x448  |      8       |      4.1ms      |   4.6ms   |      10.0ms      | 21.8ms  | 28%  |
|     8     |  448x448  |      8       |      8.7ms      |   7.0ms   |      20.3ms      | 40.7ms  | 23%  |
|    16     |  448x448  |      32      |      28.1       |   14.7    |      38.7ms      | 92.0ms  |  -   |
|    32     |  448x448  |      32      |     36.2ms      |   26.3    |      75.7ms      | 163.5ms |  -   |

note: batch size have some advantage in inference but can't speed up preprocess and postprocess.

optimize post process：

| batchsize | inputsize | maxbatchsize | preprocess time | inference | postprocess time |  all   | GPU  |
| :-------: | :-------: | :----------: | :-------------: | :-------: | :--------------: | :----: | :--: |
|     1     |  448x448  |      8       |      1.0ms      |   2.3ms   |      0.09ms      | 3.5ms  | 70%  |
|     2     |  448x448  |      8       |      2.2ms      |   2.8ms   |      0.2ms       | 5.3ms  | 60%  |
|     4     |  448x448  |      8       |      3.7ms      |   5.0ms   |      0.3ms       | 8.4ms  | 55%  |
|     8     |  448x448  |      8       |      7.5ms      |   6.5ms   |      0.67ms      | 14.9ms | 50%  |
|    16     |  448x448  |      32      |      26ms       |   13ms    |      1.3ms       |  41ms  | 40%  |
|    32     |  448x448  |      32      |      32ms       |   22ms    |      2.7ms       | 56.6ms | 50%  |

use nvidia npp library to speed up preprocess：

| batchsize | inputsize | maxbatchsize | preprocess time | inference | postprocess time |  all   | GPU  |
| :-------: | :-------: | :----------: | :-------------: | :-------: | :--------------: | :----: | :--: |
|     1     |  448x448  |      8       |      0.2ms      |   2.3ms   |      0.1ms       | 2.6ms  | 91%  |
|     2     |  448x448  |      8       |      0.3ms      |   3.0ms   |      0.2ms       | 3.5ms  | 85%  |
|     4     |  448x448  |      8       |      0.5ms      |   4.1ms   |      0.32ms      | 5.0ms  | 82%  |
|     8     |  448x448  |      8       |      1.2ms      |   6.3ms   |      0.77ms      | 8.3ms  | 79%  |
|    16     |  448x448  |      32      |      2.2ms      |   14ms    |      1.3ms       | 16.7ms | 80%  |
|    32     |  448x448  |      32      |      5.0ms      |   22ms    |      2.8ms       | 29.3ms | 77%  |


### INT8 inference
INT8 calibration table can generate by [INT8-Calibration-Tool](https://github.com/clancylian/retinaface/tree/master/INT8-Calibration-Tool).

### Accuracy

![https://raw.githubusercontent.com/clancylian/retinaface/master/data/retinaface-widerface%E6%B5%8B%E8%AF%95.png](https://raw.githubusercontent.com/clancylian/retinaface/master/data/retinaface-widerface%E6%B5%8B%E8%AF%95.png)

