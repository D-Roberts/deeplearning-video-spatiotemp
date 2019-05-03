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


import mxnet as mx
import numpy as np

from net_builder import LorenzBuilder
from data_iterator_builder import DIterators
from metric_util import rmse, plot_predictions

class Predict(object):
    """

    The modeling engine to predict with existing neural network model.
    """
    def __init__(self, config):
        self.batch_size_predict = 1
        self.model = config.model
        self.ctx = mx.cpu()
        self.dilation_depth = config.dilation_depth
        self.in_channels = config.in_channels
        self.ts = config.trajectory
        self.evals = config.evaluation
        self.checkp_path = config.checkp_path
        self.lorenz_steps = config.lorenz_steps
        self.predict_input_path = config.predict_input_path
        self.ntest = config.test_size
        # TODO: args should really be inherited from an argparse object.

    def predict(self):

        # load_model
        net = LorenzBuilder(self.dilation_depth, self.in_channels, self.ctx, self.checkp_path, for_train=False).build()
        test_data = np.loadtxt(self.predict_input_path)
        predict_data_iter = DIterators(self.batch_size_predict, self.dilation_depth, self.model,
                        self.lorenz_steps, self.ntest, self.ts, self.predict_input_path,
                        for_train=False).predict_iterator(test_data)

        # labels could be empty if pure prediction
        labels = []
        preds = []

        for x, y in predict_data_iter:
            x = x.reshape((x.shape[0], self.in_channels, -1))
            y_hat = net(x)
            preds.extend(y_hat.asnumpy().tolist()[0])
            labels.extend(y.asnumpy().tolist())

        np.savetxt('assets/predictions/preds.txt', preds)
        np.savetxt('assets/predictions/labels.txt', labels)

        if self.evals:
            rmse_test = rmse(preds, labels)
            print('rmse test', rmse_test)
            plt = plot_predictions(preds, labels)
            plt.savefig('assets/preds_w')
            plt.show()
            plt.close()





