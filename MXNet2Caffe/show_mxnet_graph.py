import mxnet as mx
import argparse


def load_model_sym(mprefix, epoch):
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=mprefix, epoch=epoch)
    return sym


def parse_args():
    parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
    parser.add_argument('mx_model', type=str, default='model_mxnet/facega,0')

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    vec = args.mx_model.split(',')
    sym = load_model_sym(vec[0], int(vec[1]))
    mx.viz.plot_network(sym)
