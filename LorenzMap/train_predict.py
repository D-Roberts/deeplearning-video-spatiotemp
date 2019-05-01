'''
Train and predict methods for Lorenz map one step ahead prediction model
with conditional and unconditional WaveNet like architecture and
conditional and unconditional LSTM architecture.

'''
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
import numpy as np
import mxnet as mx
from mxnet import gluon, init, autograd
from mxnet import ndarray as nd
import matplotlib
matplotlib.use('TkAgg')
from data_util import getDataLorenz
from data_util import get_mc_mt_gluon_iterator, get_gluon_iterator
from metric_util import plot_losses, plot_predictions
from metric_util import rmse
from models import Lorenz

np.random.seed(1234)
ctx = mx.cpu()


def train_net_SGD_gluon_mc(ts, data_x, data_y, data_z, in_channels, receptive_field, epochs, batch_size, lr, l2_reg):
    """train with SGD"""

    # pad training data with rec field length and get train iterator
    data_x = np.append(np.zeros(receptive_field), data_x, axis=0)
    data_y = np.append(np.zeros(receptive_field), data_y, axis=0)
    data_z = np.append(np.zeros(receptive_field), data_z, axis=0)

    g = get_mc_mt_gluon_iterator(data_x, data_y, data_z, receptive_field=receptive_field, shuffle=True,
                           batch_size=batch_size, last_batch='discard')

    # build model
    net = Lorenz(r=receptive_field, in_channels=in_channels, L=4, k=2, M=1)
    net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': l2_reg})
    loss = gluon.loss.L1Loss() # works better
    # loss = gluon.loss.L2Loss()

    loss_save = []
    best_loss = sys.maxsize

    for epoch in range(epochs):
        total_epoch_loss, nb = 0, 0

        for x, y in g:
            nb += 1
            with autograd.record():
                y_hat = net(x)
                l = loss(y_hat, y[:,ts])
                total_epoch_loss += nd.sum(l).asscalar()

            l.backward()
            trainer.step(batch_size, ignore_stale_grad=True)

            # Inspect gradients
            grads = [i.grad(ctx) for i in net.collect_params().values() if i._grad is not None]

        # epoch loss on train set
        current_loss = total_epoch_loss/nb
        loss_save.append(current_loss)
        print('Epoch {}, loss {}'.format(epoch, current_loss))

        if current_loss < best_loss:
            best_loss = current_loss
            net.save_params('assets/best_model_x')

        print('best epoch loss: ', best_loss)

    # return best model
    new_net = Lorenz(r=receptive_field, in_channels=in_channels, L=4, k=2, M=1)
    new_net.load_params('assets/best_model_x', ctx=ctx)
    return loss_save, new_net

def train_net_SGD_gluon(data, in_channels, receptive_field, epochs, batch_size, lr, l2_reg):
    """train with SGD"""

    # pad training data with rec field length and get train iterator
    data = np.append(np.zeros(receptive_field), data, axis=0)
    g = get_gluon_iterator(data, receptive_field=receptive_field, shuffle=True,
                           batch_size=batch_size, last_batch='discard')

    # build model
    net = Lorenz(r=receptive_field, L=4, k=2, M=1)
    net.collect_params().initialize(mx.init.Xavier(magnitude=2, rnd_type='gaussian', factor_type='in'), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr, 'wd': l2_reg})
    loss = gluon.loss.L1Loss()

    loss_save = []
    best_loss = sys.maxsize
    for epoch in range(epochs):

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
            trainer.step(batch_size, ignore_stale_grad=True)

        current_loss = total_epoch_loss/nb
        loss_save.append(current_loss)
        print('Epoch {}, loss {}'.format(epoch, current_loss))

        if current_loss < best_loss:
            best_loss = current_loss
            net.save_params('assets/best_model_w')

        print('best epoch loss: ', best_loss)

    # return best model
    new_net = Lorenz(r=receptive_field, L=4, k=2, M=1)
    new_net.load_params('assets/best_model_w', ctx=ctx)
    return loss_save, new_net


def predict(data_iter, in_channels, net, ts):
    '''net is the trained net. Which ts to predict for is ts=0, 1, or 2'''

    labels = []
    preds = []

    for X, y in data_iter:
        X = X.reshape((X.shape[0], in_channels, -1))
        y_hat = net(X)
        preds.extend(y_hat.asnumpy().tolist()[0])
        labels.extend(y[:,ts].asnumpy().tolist())

    return preds, labels

def train_predict_cw(ts=0, ntest=500, Lorenznsteps=1500, batch_size=32, epochs=100):

    x, y, z = getDataLorenz(Lorenznsteps)

    nTest = ntest
    nTrain = len(x) - nTest
    train_x, test_x = x[:nTrain], x[nTrain:]
    train_y, test_y = y[:nTrain], y[nTrain:]
    train_z, test_z = z[:nTrain], z[nTrain:]

    ts = ts
    batch_size = batch_size
    losses, net = train_net_SGD_gluon_mc(ts, train_x, train_y, train_z, in_channels=3, receptive_field=16,
                                         batch_size=batch_size, epochs=epochs, lr=0.001, l2_reg=0.001)

    # Plot losses
    plt = plot_losses(losses, 'cw')
    # plt.show()
    plt.savefig('assets/losses_cw')
    plt.close()

    # Make predictions on test set
    batch_size = 1
    receptive_field = 16
    in_channels = 3
    data_x = test_x
    data_y = test_y
    data_z = test_z

    g = get_mc_mt_gluon_iterator(data_x, data_y, data_z, receptive_field=receptive_field, shuffle=False,
                                 batch_size=batch_size, last_batch='discard')

    preds, labels = predict(g, in_channels, net, ts)
    rmse_test = rmse(preds, labels)
    print('rmse test', rmse_test)
    plt = plot_predictions(preds[:100], labels[:100])
    plt.savefig('assets/preds_cwn')
    plt.close()

def train_predict_w(ts=0, ntest=500, Lorenznsteps=1500, batch_size=32, epochs=100):
    '''Training univariate trajectory.

    '''

    x, y, z = getDataLorenz(Lorenznsteps)

    nTest = ntest
    nTrain = len(x) - nTest
    train_x, test_x = x[:nTrain], x[nTrain:]
    train_y, test_y = y[:nTrain], y[nTrain:]
    train_z, test_z = z[:nTrain], z[nTrain:]

    if ts == 0:
        train_data, test_data = train_x, test_x
    elif ts == 1:
        train_data, test_data = train_y, test_y
    elif ts == 2:
        train_data, test_data = train_z, test_z

    losses, net = train_net_SGD_gluon(train_data, in_channels=1, receptive_field=16,
                                      batch_size=batch_size, epochs=epochs, lr=0.001, l2_reg=0.001)

    # Plot losses
    plt = plot_losses(losses, 'w')
    # plt.show()
    plt.savefig('assets/losses_w')
    plt.close()

    # Make predictions on test set
    batch_size = 1
    receptive_field = 16

    g = get_gluon_iterator(test_data, receptive_field=receptive_field, shuffle=False,
                           batch_size=batch_size, last_batch='discard')

    preds, labels = predict(g, 1, net, ts)
    rmse_test = rmse(preds, labels)
    print('rmse test', rmse_test)
    plt = plot_predictions(preds[:100], labels[:100])
    plt.savefig('assets/preds_w')
    plt.close()
