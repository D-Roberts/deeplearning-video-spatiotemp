"""Bare bones training script for i3d.

Goal is to train with Kinetics dataset to replicate Tf results.

For now - train with 1 example to establish pipeline.
"""

from multiprocessing import cpu_count

import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler


import i3d


_SAVE_DIR = 'assets/data/checkp_gluon'


_SAMPLE_PATHS = {
    'rgb': 'assets/data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'assets/data/v_CricketShot_g04_c01_flow.npy',
}

_LABEL_MAP_PATH = 'assets/data/label_map.txt'

ctx = mx.cpu()
_IMAGE_SIZE = 224
_NUM_CLASSES = 4
_SAMPLE_VIDEO_FRAMES = 79
_BATCH_SIZE = 1
_NUM_CHANNELS = 3


def main():

    batch_size = 1
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)][:4]

    num_training_samples = 1
    context = mx.cpu()
    num_batches = num_training_samples // batch_size
    optimizer = mx.optimizer.Adam()
    # default params
    # lr = 0.001
    epochs = 1


    def get_data():
        # only 1 example
        X = mx.nd.array(np.load(_SAMPLE_PATHS['rgb']), ctx=ctx)
        # hard code the label as having first class correct

        # reshape to work with gluon expectations
        X = X.reshape((_BATCH_SIZE, _NUM_CHANNELS, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE))

        y = mx.nd.array([1]).reshape((1,-1))
        dataset = gluon.data.ArrayDataset(X, y)
        data_loader = gluon.data.DataLoader(dataset, batch_size, last_batch='keep')
        return data_loader

    def train(context=ctx):
        net = i3d.i3d()
        net.initialize()
        trainer = gluon.Trainer(net.collect_params(), optimizer)

        loss = gluon.loss.SoftmaxCrossEntropyLoss()
        train_metric = mx.metric.Accuracy()
        train_data = get_data()

        for epoch in range(epochs):
            train_metric.reset()

            for x, y in train_data:

                with ag.record():
                    output = net(x)
                    l = loss(output, y)

                l.backward()
                trainer.step(batch_size)

            # accuracy at 1
            train_metric.update(y, output)
            net.save_parameters(os.path.join(_SAVE_DIR, 'first'))


    train(context)


if __name__ == '__main__':
    main()
