try:
    import caffe
except ImportError:
    import os, sys

    curr_path = os.path.abspath(os.path.dirname(__file__))
    sys.path.append("/usr/local/caffe/python")
    import caffe
