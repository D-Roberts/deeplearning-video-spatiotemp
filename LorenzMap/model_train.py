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
"""
Trainer class.
"""
import sys

import mxnet as mx
from mxnet import autograd, gluon, nd
import numpy as np
from tqdm import trange

from models import Lorenz
from data_util import generate_synthetic_lorenz, get_gluon_iterator


# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, no-member, no-self-use

# set gpu count TODO

class Train(object):
    """
    Training engine for Lorenz architecture.
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.dilation_depth = config.dilation_depth
        self.lorenz_steps = config.lorenz_steps
        self.ntest = config.test_size
        self.ts = config.trajectory
        self.ctx = mx.cpu()
        self.lr = config.learning_rate
        self.l2_reg = config.l2_regularization

    def save_model(self, epoch, current_loss):
        """

        :param epoch:
        :param current_loss:
        :return:
        """
        filename = 'models/best_perf_epoch_' + str(epoch) + "_loss_" + str(current_loss)
        self.net.save_params(filename)

    def train(self):
        """

        :return:
        """
        x, y, z = generate_synthetic_lorenz(self.lorenz_steps)
        print(x)

        nTrain = len(x) - self.ntest
        train_x, test_x = x[:nTrain], x[nTrain:]
        train_y, test_y = y[:nTrain], y[nTrain:]
        train_z, test_z = z[:nTrain], z[nTrain:]

        if self.ts == 0:
            train_data, test_data = train_x, test_x
        elif self.ts == 1:
            train_data, test_data = train_y, test_y
        elif self.ts == 2:
            train_data, test_data = train_z, test_z

        receptive_field = 16
        in_channels = 1

        data = np.append(np.zeros(receptive_field), train_data, axis=0)

        g = get_gluon_iterator(train_data, receptive_field=receptive_field, shuffle=True,
                               batch_size=self.batch_size, last_batch='discard')

        # build model
        net = Lorenz(L=4, k=2, M=1)
        net.collect_params().initialize(mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'), ctx=self.ctx)
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.l2_reg})
        loss = gluon.loss.L1Loss()

        loss_save = []
        best_loss = sys.maxsize
        for epoch in trange(self.epochs):

            # Inspect params
            params = net.collect_params()
            # print(params)
            total_epoch_loss, nb = 0, 0

            for x, y in g:
                # number of batches
                nb += 1
                with autograd.record():
                    # (batch_sizeXin_channelsXwidth)
                    x = x.reshape((x.shape[0], in_channels, x.shape[1]))
                    y_hat = net(x)
                    l = loss(y_hat, y)
                    total_epoch_loss += nd.sum(l).asscalar()

                l.backward()
                trainer.step(self.batch_size, ignore_stale_grad=True)

            current_loss = total_epoch_loss / nb
            loss_save.append(current_loss)
            print('Epoch {}, loss {}'.format(epoch, current_loss))

            if current_loss < best_loss:
                best_loss = current_loss
                net.save_params('assets/best_model_w')

            print('best epoch loss: ', best_loss)
