"""
Generate the Lorenz data for toy example in:
https://arxiv.org/pdf/1703.04691.pdf
Generate t=1,1500 X, Y and Z as three time series.

Get data iterators.
"""

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

ctx = mx.cpu()
np.random.seed(1234)

def getDataLorenz(stepCnt, dt = 0.01, initx = 0., inity = 1., initz = 1., s = 5, r = 20, b = 2):
    '''Generate Lorenz map via Euler'''
    xs = np.zeros(stepCnt+1)
    ys = np.zeros(stepCnt+1)
    zs = np.zeros(stepCnt+1)
    xs[0], ys[0], zs[0] = (initx, inity, initz)
    for i in range(stepCnt):
        x_dot = s*(ys[i] - xs[i])
        y_dot = r*xs[i] - ys[i] - xs[i]*zs[i]
        z_dot = xs[i]*ys[i] - b*zs[i]
        xs[i+1] = xs[i] + (x_dot * dt)
        ys[i+1] = ys[i] + (y_dot * dt)
        zs[i+1] = zs[i] + (z_dot * dt)
    # Rescale data to [-0.5, 0.5] range
    xs = (xs - np.amax(xs))/(np.amax(xs)-np.amin(xs)) + 0.5
    ys = (ys - np.amax(ys))/(np.amax(ys)-np.amin(ys)) + 0.5
    zs = (zs - np.amax(zs))/(np.amax(zs)-np.amin(zs)) + 0.5
    return xs, ys, zs

def get_gluon_iterator(data, receptive_field, batch_size, shuffle=True, last_batch='discard'):
    '''Input is either the train set, the validation set or the test set,
    with padding already done if necessary.
    Predict 1 step ahead, 1 time series, x length is receptive field.
    '''

    T = data.shape[0]
    X = nd.zeros((T-receptive_field, receptive_field))
    y = nd.zeros((T-receptive_field, 1))

    for i in range(T-receptive_field):
        X[i, :] = data[i:i+receptive_field]
        y[i] = data[i+receptive_field]

    dataset = gluon.data.ArrayDataset(X, y)
    diter = gluon.data.DataLoader(dataset, batch_size, shuffle=shuffle, last_batch=last_batch)
    return diter

def get_mc_mt_gluon_iterator(data_x, data_y, data_z, receptive_field, batch_size, shuffle=True, last_batch='discard'):
    '''Multi channel multi task conditional iterator, 3 output, 1 step ahead.
    '''

    T = data_x.shape[0]
    Xx = nd.zeros((T-receptive_field, receptive_field))
    Xy = nd.zeros((T-receptive_field, receptive_field))
    Xz = nd.zeros((T-receptive_field, receptive_field))
    y = nd.zeros((T-receptive_field, 3))

    for i in range(T-receptive_field):
        Xx[i, :] = data_x[i:i+receptive_field]
        Xy[i, :] = data_y[i:i + receptive_field]
        Xz[i, :] = data_z[i:i + receptive_field]
        # labels
        y[i, 0] = data_x[i+receptive_field]
        y[i, 1] = data_y[i + receptive_field]
        y[i, 2] = data_z[i + receptive_field]

    X3 = nd.zeros((T-receptive_field, 3, receptive_field))
    for i in range(X3.shape[0]):
        X3[i,:,:]= nd.concatenate([Xx[i, :].reshape((1,-1)), Xy[i, :].reshape((1,-1)), Xz[i, :].reshape((1,-1))])

    dataset = gluon.data.ArrayDataset(X3, y)
    diter = gluon.data.DataLoader(dataset, batch_size, shuffle=shuffle, last_batch=last_batch)
    return diter
