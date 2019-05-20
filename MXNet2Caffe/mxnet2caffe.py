import sys, argparse
import find_mxnet, find_caffe
import mxnet as mx
import caffe
import json

parser = argparse.ArgumentParser(description='Convert MXNet model to Caffe model')
parser.add_argument('--mx-model',    type=str, default='model_mxnet/mnet.25')
parser.add_argument('--mx-epoch',    type=int, default=0)
parser.add_argument('--cf-prototxt', type=str, default='model_caffe/mnet.25.prototxt')
parser.add_argument('--cf-model',    type=str, default='model_caffe/mnet.25.caffemodel')
args = parser.parse_args()

# ------------------------------------------
# Load
_, arg_params, aux_params = mx.model.load_checkpoint(args.mx_model, args.mx_epoch)
net = caffe.Net(args.cf_prototxt, caffe.TRAIN)   

mx_json = '%s-symbol.json' % args.mx_model
with open(mx_json) as json_file:
  jdata = json.load(json_file)

# ------------------------------------------
# Convert

# python3
param = arg_params.copy()
param.update(aux_params)
all_keys = sorted(param)

# python2
# all_keys = arg_params.keys() + aux_params.keys()
# all_keys.sort()

print('----------------------------------\n')
print('ALL KEYS IN MXNET:')
print(all_keys)
print('%d KEYS' %len(all_keys))

print('----------------------------------\n')
print('VALID KEYS:')
for i_key,key_i in enumerate(all_keys):
  try:    
    if 'data' is key_i:
      pass
    elif '_weight' in key_i:
      key_caffe = key_i.replace('_weight','')
      if 'fc' in key_i:
        print(key_i)
        print(arg_params[key_i].shape)
        print(net.params[key_caffe][0].data.shape)
      if key_caffe not in net.params:
        key_caffe = key_caffe + '_fwd'
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat    
    elif '_bias' in key_i:
      key_caffe = key_i.replace('_bias','')
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat   
    elif '_gamma' in key_i and 'relu' not in key_i:               
      # for mxnet batchnorm layer, if fix_gamma == 'True', the values should be 1.
      fix_gamma_param = False
      for layer in jdata['nodes']:
        if layer['name'] == key_i:
          if 'attrs' in layer and 'fix_gamma' in layer['attrs'] and str(layer['attrs']['fix_gamma']) == 'True':
            fix_gamma_param = True  
          else:
            fix_gamma_param = False
          break
      key_caffe = key_i.replace('_gamma', '_scale')
      if key_caffe not in net.params:
        key_caffe = key_i.replace('_gamma', '_fwd_scale')

      if fix_gamma_param:
        net.params[key_caffe][0].data[...] = 1
      else:
        net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat

      # key_caffe = key_i.replace('_gamma', '_scale')
      # if key_caffe not in net.params:
      #   key_caffe = key_i.replace('_gamma', '_fwd_scale')

      # print("{}: {}->{}".format(key_i, arg_params[key_i].shape, net.params[key_caffe][0].data.shape))
      # net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat

    # TODO: support prelu
    elif '_gamma' in key_i and 'relu' in key_i:  # for prelu
      key_caffe = key_i.replace('_gamma', '')
      assert (len(net.params[key_caffe]) == 1)
      net.params[key_caffe][0].data.flat = arg_params[key_i].asnumpy().flat
    elif '_beta' in key_i:
      key_caffe = key_i.replace('_beta', '_scale')
      if key_caffe not in net.params:
        key_caffe = key_i.replace('_beta', '_fwd_scale')
      
      print(arg_params[key_i].asnumpy().flat)
      net.params[key_caffe][1].data.flat = arg_params[key_i].asnumpy().flat    
    elif '_moving_mean' in key_i:
      key_caffe = key_i.replace('_moving_mean','')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat 
      net.params[key_caffe][2].data[...] = 1 
    elif '_running_mean' in key_i:
      key_caffe = key_i.replace('_running_mean','_fwd')
      net.params[key_caffe][0].data.flat = aux_params[key_i].asnumpy().flat 
      net.params[key_caffe][2].data[...] = 1 
    elif '_moving_var' in key_i:
      key_caffe = key_i.replace('_moving_var','')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat    
      net.params[key_caffe][2].data[...] = 1 
    elif '_running_var' in key_i:
      key_caffe = key_i.replace('_running_var','_fwd')
      net.params[key_caffe][1].data.flat = aux_params[key_i].asnumpy().flat    
      net.params[key_caffe][2].data[...] = 1 
    else:
      sys.exit("Warning!  Unknown mxnet:{}".format(key_i))
  
    print("% 3d | %s -> %s, initialized." 
           %(i_key, key_i.ljust(40), key_caffe.ljust(30)))
    
  except KeyError:
    print("\nWarning!  key error mxnet:{}".format(key_i))  
      
# ------------------------------------------
# Finish
net.save(args.cf_model)
print("\n- Finished.\n")








