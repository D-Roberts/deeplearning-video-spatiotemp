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
from mxnet import ndarray as nd
import math
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def rmse(preds, labels):
    '''RMSE metric reported in literature
    '''
    mse = mx.metric.MSE()
    mse.update(labels=nd.array(labels), preds=nd.array(preds))
    return math.sqrt(mse.get()[1])

def mase(preds_model, preds_naive, labels):
    '''Mean absolute scaled arror to compare model performance to naive forecast
    '''
    mae_model = np.mean(np.abs(np.array(preds_model)-np.array(labels)))
    mae_naive = np.mean(np.abs(np.array(preds_naive)-np.array(labels)))
    return mae_model/mae_naive

def mae(preds, labels):
    '''Mean absolute error
    '''
    mae = mx.metric.MAE()
    mae.update(labels=nd.array(labels), preds=nd.array(preds))
    return mae.get()[1]

def plot_losses(losses, label):
    '''Plot losses per epoch. Train or validation loss or
    another metric.
    '''
    x_axis = np.linspace(0, len(losses), len(losses), endpoint=True)
    plt.semilogy(x_axis, losses, label=label)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    return plt

def plot_predictions(preds, labels):
    '''Plot predictions vs ground truth.
    '''
    T = len(preds)
    time = nd.arange(0, T)
    plt.plot(time.asnumpy(), labels, label='labels')
    plt.plot(time.asnumpy(), preds, label='predictions')
    plt.legend()
    return plt