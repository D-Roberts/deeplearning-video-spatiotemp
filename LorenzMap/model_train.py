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

from net_builder import LorenzBuilder
from data_iterator_builder import DIterators
from metric_util import plot_losses
# pylint: disable=invalid-name, too-many-arguments, too-many-instance-attributes, no-member, no-self-use

# set gpu count TODO

class Train(object):
    """
    Training engine for Lorenz architecture.
    """

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.in_channels = config.in_channels
        self.dilation_depth = config.dilation_depth
        self.lorenz_steps = config.lorenz_steps
        self.ntest = config.test_size
        self.ts = config.trajectory
        self.ctx = mx.cpu()
        self.lr = config.learning_rate
        self.l2_reg = config.l2_regularization
        self.plot_losses = config.plot_losses
        self.checkp_path = config.checkp_path
        self.model = config.model
        self.predict_input_path = config.predict_input_path


    def save_model(self, net):
        net.save_params(self.checkp_path)

    def train(self):

        net = LorenzBuilder(self.dilation_depth, self.in_channels, self.ctx, self.checkp_path, for_train=True).build()
        trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': self.lr, 'wd': self.l2_reg})
        loss = gluon.loss.L1Loss()

        g = DIterators(self.batch_size, self.dilation_depth, self.model,
                       self.lorenz_steps, self.ntest, self.ts, self.predict_input_path,
                       for_train=True).train_iterator()

        loss_save = []
        best_loss = sys.maxsize

        for epoch in trange(self.epochs):
            total_epoch_loss, nb = 0, 0
            for x, y in g:
                # x shape: (batch_sizeXin_channelsXwidth)
                x = x.reshape((self.batch_size, self.in_channels, -1))
                # print(x.shape)
                nb += 1
                with autograd.record():
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
                self.save_model(net)
            print('best epoch loss: ', best_loss)

        if self.plot_losses:
            plt = plot_losses(loss_save, 'w')
            plt.show()
            plt.savefig('assets/losses_w')
            plt.close()

