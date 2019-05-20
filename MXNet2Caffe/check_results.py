import argparse

from predictor_caffe import PredictorCaffe
from predictor_mxnet import PredictorMxNet
import numpy as np


def compare_diff_sum(tensor1, tensor2):
    pass


def compare_cosin_dist(tensor1, tensor2):
    pass


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compare_models(prefix_mxnet, prefix_caffe, size, layer_name):
    netmx = PredictorMxNet(prefix_mxnet, 0, size, internal_layer=layer_name)

    model_file = prefix_caffe + ".prototxt"
    pretrained_file = prefix_caffe + ".caffemodel"
    netcaffe = PredictorCaffe(model_file, pretrained_file, size)

    tensor = np.ones(size, dtype=np.float32)
    out_mx = netmx.forward(tensor)
    print('out_mx')
    print(out_mx)

    # tensor = (tensor - 127.5) * 0.0078125  # for most mxnet model, it is a default normalization
    netcaffe.forward(tensor)
    out_caffe = netcaffe.blob_by_name(layer_name)
    print('out_caffe')
    print(out_caffe.data)

    print('done')


def parse_args():
    parser = argparse.ArgumentParser(description='check the caffe model')
    parser.add_argument('--prefix_mxnet', type=str, default='model_mxnet/mnet.25')
    parser.add_argument('--prefix_caffe', type=str, default='model_caffe/mnet.25')
    parser.add_argument('--size', nargs='+', type=int, help='the input size of model with batch size 1',
                        default=[1, 3, 640, 640])
    parser.add_argument('--layer', type=str, default='fc1')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    prefix_mxnet = args.prefix_mxnet
    prefix_caffe = args.prefix_caffe
    size = args.size
    layer_name = args.layer

    compare_models(prefix_mxnet, prefix_caffe, size, layer_name)
