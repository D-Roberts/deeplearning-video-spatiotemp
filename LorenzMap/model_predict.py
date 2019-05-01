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

class Predict(object):
    """

    The modeling engine to predict with existing neural network model.
    """
    def __init__(self, config):
        self.batch_size_predict = 1
        self.in_channels = config.in_channels
        self.dilation_depth = config.dilation_depth
        self.lorenz_steps = config.lorenz_steps
        self.ts = config.trajectory
        self.ctx = mx.cpu()
        self.lr = config.learning_rate
        self.l2_reg = config.l2_regularization
        self.ts = config.trajectory
        self.build_model()

    # TODO: need better solution than to repeat build model. Module manager likely.
    # TODO: also argparse class.

    def build_model(self):
        """

        :return:
        """
        self.net = Lorenz(L=self.dilation_depth, in_channels=self.in_channels, k=2, M=1)
        self.net.collect_params().initialize(mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'),
                                        ctx=self.ctx)
        self.trainer = gluon.Trainer(self.net.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.l2_reg})
        self.loss = gluon.loss.L1Loss()


    def predict(self):

        receptive_field = 2 ** self.dilation_depth

        # load predict input
        # TODO: hard coded to be fixed
        test_data = np.loadtxt('/Users/denisaroberts/PycharmProjects/Lorenz/LorenzMap/assets/predictions/test.txt')

        # load_model
        self.net.load_params('assets/best_perf_model', ctx=self.ctx)

        g = get_gluon_iterator(test_data, receptive_field=receptive_field, shuffle=False,
                               batch_size=self.batch_size_predict, last_batch='discard')

        # labels could be empty
        labels = []
        preds = []

        for X, y in g:
            X = X.reshape((X.shape[0], self.in_channels, -1))
            y_hat = self.net(X)
            preds.extend(y_hat.asnumpy().tolist()[0])
            labels.extend(y[:, self.ts].asnumpy().tolist())

        np.savetxt('assets/predictions/preds.txt', preds)
        np.savetxt('assets/predictions/labels.txt', labels)


