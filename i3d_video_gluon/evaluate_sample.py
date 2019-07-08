"""Evaluate pretrained i3d model in gluon.

"""

import os

import numpy as np
import mxnet as mx
from mxnet import ndarray as nd
import gluon

import i3d


_SAMPLE_PATHS = {
    'rgb': 'assets/data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'assets/data/v_CricketShot_g04_c01_flow.npy',
}

_LABEL_MAP_PATH = 'assets/data/label_map.txt'
_SAVE_DIR = 'assets/data/checkp_gluon'

ctx = mx.cpu()
_IMAGE_SIZE = 224
_NUM_CLASSES = 400
_SAMPLE_VIDEO_FRAMES = 79
_BATCH_SIZE = 1
_NUM_CHANNELS = 3

def _test_model(net, ctx, x):
    net.initialize()
    net.collect_params().reset_ctx(ctx)
    output = net(x)
    output_softmax = nd.SoftmaxActivation(output)

def main():
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    # Test rgb model
    # rgb input has 3 channels

    # sample input
    x = mx.nd.array(np.load(_SAMPLE_PATHS['rgb']), ctx=ctx)
    # (1, 79, 224, 224, 3)

    # build model
    net = i3d.i3d()

    # load trained parameters
    net.load_parameters(os.path.join(_SAVE_DIR, 'first'))
    output = net(x)

    # get predicted top 1 class by softmax probability
    output_softmax = nd.SoftmaxActivation(output).asnumpy()[0]
    sorted_indeces = np.argsort(output_softmax)[::-1][0]

    print(kinetics_classes[sorted_indeces])


if __name__ == '__main__':
    main()



