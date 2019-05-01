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
        self.ctx = mx.cpu()

    # TODO: the trajectory should not be given  in predict
    # the iterator should be ready for the right prediction task
    # TODO: the test size nlorenz steps should not be given here
    # that has to do with get data iterator
    # if possible also shaping in the iterator should be ready to go.

    # TODO: selection of model (w/cw) and target (x, y, z) should be made
    # outside train and predict in data iterator build


    def predict(self, module_manager, predict_data_iter):

        # load_model
        net = module_manager.build()
        net.load_params('assets/best_perf_model', ctx=self.ctx)

        # labels could be empty
        labels = []
        preds = []

        for X, y in g:
            X = X.reshape((X.shape[0], self.in_channels, -1))
            y_hat = net(X)
            preds.extend(y_hat.asnumpy().tolist()[0])
            labels.extend(y[:, self.ts].asnumpy().tolist())

        np.savetxt('assets/predictions/preds.txt', preds)
        np.savetxt('assets/predictions/labels.txt', labels)


