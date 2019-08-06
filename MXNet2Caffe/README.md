

## 模型转换工具
参考[MXNet2Caffe](https://github.com/cypw/MXNet2Caffe)

需要自己添加一些层，caffe中没有upsample层，使用deconvition替代，会有精度一点损失。

### 支持的层有：

- Convolution
- ChannelwiseConvolution
- BatchNorm
- Activation
- ElementWiseSum
- _Plus
- elemwise_add
- Concat
- Pooling
- Flatten
- FullyConnected
- SoftmaxActivation
- Reshape
- UpSampling
- Crop

