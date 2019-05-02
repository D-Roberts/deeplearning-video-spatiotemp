# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import sys

import mxnet as mx
from mxnet import autograd, gluon, nd
import numpy as np
from tqdm import trange

from models import Lorenz
from data_util import generate_synthetic_lorenz, get_gluon_iterator
from net_builder import LorenzBuilder
from data_iterator_builder import DIterators

class Predict(object):
    """

    The modeling engine to predict with existing neural network model.
    """
    def __init__(self, config):
        self.batch_size_predict = 1
        self.ctx = mx.cpu()
        self.dilation_depth = config.dilation_depth
        self.in_channels = config.in_channels
        self.ts = config.trajectory
        self.evals = config.evaluation
        self.checkp_path = config.checkp_path
        # TODO: args should really be inherited from an argparse object.
        # lots of repetitions of multiple arguments

    def predict(self):

        # load_model
        net = LorenzBuilder(self.dilation_depth, self.in_channels, self.ctx, self.checkp_path, for_train=False)

        # TODO: load a test file from a location given in argparse
        test_data = np.loadtxt('/Users/denisaroberts/PycharmProjects/Lorenz/LorenzMap/assets/predictions/test.txt')

        # iterator here
        predict_data_iter = DIterators(for_train=False).predict_iterator(test_data)

        # labels could be empty if pure prediction
        labels = []
        preds = []

        for X, y in predict_data_iter:
            X = X.reshape((X.shape[0], self.in_channels, -1))
            y_hat = net(X)
            preds.extend(y_hat.asnumpy().tolist()[0])
            labels.extend(y[:, self.ts].asnumpy().tolist())

        np.savetxt('assets/predictions/preds.txt', preds)
        np.savetxt('assets/predictions/labels.txt', labels)

        if self.evals:
            # predictions evaluations
            rmse_test = rmse(preds, labels)
            print('rmse test', rmse_test)
            plt = plot_predictions(preds, labels)
            # plt.savefig('assets/preds_w')
            plt.show()
            plt.close()





