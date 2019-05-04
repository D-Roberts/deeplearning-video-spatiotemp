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

from mxnet import ndarray as nd
from mxnet import gluon
import numpy as np

from data_util import generate_synthetic_lorenz

class DIterators(object):
    def __init__(self, batch_size, dilation_depth, model, LorenzSteps, test_size,
                 trajectory, predict_input_path, for_train=True):
        self.batch_size = batch_size
        self.ts = trajectory
        self.model = model
        self.stepcount = LorenzSteps
        self.ntest = test_size
        self.for_train = for_train
        self.receptive_field = dilation_depth ** 2
        self.predict_input_path = predict_input_path

    def build_iterator(self, data):
        T = data.shape[0]
        X3 = nd.zeros((T - self.receptive_field, data.shape[1], self.receptive_field))
        y = nd.zeros((T - self.receptive_field, data.shape[1]))

        for i in range(T - self.receptive_field):
            for j in range(data.shape[1]):
                X3[i, j, :] = data[i:i + self.receptive_field, j]
                y[i, j] = data[i + self.receptive_field, j]

        if self.model == 'cw':
            dataset = gluon.data.ArrayDataset(X3, y[:, self.ts])
        else:
            dataset = gluon.data.ArrayDataset(X3[:, self.ts, :], y[:, self.ts])

        if self.for_train:
            diter = gluon.data.DataLoader(dataset, self.batch_size, shuffle=True, last_batch='discard')
        else:
            diter = gluon.data.DataLoader(dataset, self.batch_size, shuffle=False, last_batch='keep')
        return diter

    def predict_iterator(self, predict_input):
        return self.build_iterator(predict_input)

    def train_iterator(self):
        data = generate_synthetic_lorenz(self.stepcount)
        nTrain = data.shape[0] - self.ntest
        train_data, test_data = data[:nTrain, :], data[nTrain:, :]
        train_data = np.append(np.zeros((self.receptive_field, train_data.shape[1])), train_data, axis=0)
        np.savetxt(self.predict_input_path, test_data)
        return self.build_iterator(train_data)



