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
    def __init__(self, options):
        self._options = options
        self.ctx = mx.cpu()

    def predict(self):

        # load_model
        net = LorenzBuilder(self._options.dilation_depth, self._options.in_channels, self.ctx,
                            self._options.check_path, for_train=False).build()

        test_data = np.loadtxt(self._options.predict_input_path)

        predict_data_iter = DIterators(self._options.batch_size_predict,
                                       self._options.dilation_depth, self._options.model,
                       self._options.lorenz_steps, self._options.test_size, self._options.trajectory,
                       self._options.predict_input_path,
                       for_train=False).predict_iterator(test_data)

        # labels could be empty if pure prediction
        labels = []
        preds = []

        for x, y in predict_data_iter:
            x = x.reshape((x.shape[0], self._options.in_channels, -1))
            y_hat = net(x)
            preds.extend(y_hat.asnumpy().tolist()[0])
            labels.extend(y.asnumpy().tolist())

        np.savetxt('assets/predictions/preds.txt', preds)
        np.savetxt('assets/predictions/labels.txt', labels)

        if self._options.evaluation:
            rmse_test = rmse(preds, labels)
            print('rmse test', rmse_test)
            plt = plot_predictions(preds, labels)
            plt.savefig('assets/preds_w')
            plt.show()
            plt.close()





