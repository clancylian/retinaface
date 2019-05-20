import mxnet as mx
from collections import namedtuple

Batch = namedtuple('Batch', ['data'])


class PredictorMxNet:
    def __init__(self, mprefix, epoch, size, ctx=mx.cpu(), internal_layer='fc1'):
        self.size = size
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix=mprefix,
                                                               epoch=epoch)

        internal_layer += '_output'

        # Get the output of intermediate layer
        if internal_layer != 'fc1_output' and internal_layer in sym.get_internals().list_outputs():
            internals = sym.get_internals()
            temp_layer = internals[internal_layer]
            sym = mx.symbol.Group([sym, temp_layer])

        self.mod = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        self.mod.bind(for_training=False, data_shapes=[('data', size)], label_shapes=self.mod._label_shapes)
        self.mod.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)

    def forward(self, tensor):
        self.mod.forward(Batch([mx.nd.array(tensor)]))
        out = self.mod.get_outputs()
        return out
