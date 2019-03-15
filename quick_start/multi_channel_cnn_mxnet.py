"""
Time series multichannel multistep ahead prediction.

Example: Predict 7 steps ahead using multichannel
(8 time series as inputs) and 14 time steps CNN.

Dataset: public energy consumption data.
"""


import os
import logging

logging.getLogger().setLevel(logging.DEBUG)

import numpy as np
from numpy import array
import pandas as pd
from numpy import array


from data_utils_ts import *

data_dir = "data"
ctx = mx.cpu()

def get_train_test_iterators():

    data = pd.read_csv(os.path.join(data_dir,
                                    'household_power_consumption_days.csv'), header=0,
                       infer_datetime_format=True,
                       parse_dates=['datetime'], index_col=['datetime'])

    print(data.head())
    data = data.values

    # split into train and test
    train, test = split_dataset(data)

    # get inputs and outputs
    # use 14 steps in the past and forecast next 7
    train_x, train_y = to_supervised(train, n_input=14, n_out=7)
    print("data shape", train_x.shape)
    # shape is (1092, 14, 8)
    test_x, test_y = to_supervised(test, n_input=14, n_out=7)

    epochs, batch_size = 70, 16

    n_timesteps, n_features, n_outputs = train_x.shape[1], \
                                         train_x.shape[2], \
                                         train_y.shape[1]

    train_it = mx.io.NDArrayIter(data=train_x,
                                 label=train_y,
                                 batch_size=batch_size,
                                 label_name='lin_reg')

    test_it = mx.io.NDArrayIter(data=train_x,
                                label=train_y,
                                batch_size=batch_size,
                                label_name='lin_reg')

    return train_it, test_it


def build_network(n_outputs=7):

    X = mx.sym.Variable('data')
    Y = mx.sym.Variable('lin_reg')

    net = mx.sym.Convolution(data=X, kernel=3, num_filter=32)
    net = mx.sym.Activation(net, act_type='relu')
    net = mx.sym.Convolution(net, num_filter=32, kernel=3)
    net = mx.sym.Activation(net, act_type='relu')
    net = mx.sym.Pooling(net, pool_type='max', kernel=2)
    net = mx.sym.Convolution(net, num_filter=16, kernel=3)
    net = mx.sym.Activation(net, act_type='relu')

    # flatten before dense
    net = net.flatten()

    # dense and then output layer
    net = mx.sym.FullyConnected(data=net, num_hidden=100)
    net = mx.sym.Activation(net, act_type='relu')
    net = mx.sym.FullyConnected(data=net, num_hidden=n_outputs)

    # and regression output layer
    net = mx.sym.LinearRegressionOutput(data=net, label=Y)
    return net


def main():

    # Train and dev iterators

    train_it, test_it = get_train_test_iterators()

    # Train
    net = build_network()
    model = mx.mod.Module(symbol=net,
                          context=ctx,
                          data_names=['data'],
                          label_names=['lin_reg'])

    model.fit(train_it,
              test_it,
              num_epoch=epochs,
              optimizer='adam',
              optimizer_params={'learning_rate':0.001},
              eval_metric='rmse',
              batch_end_callback=mx.callback.Speedometer(batch_size, 100))


    # Score model on the dev set
    metric = mx.metric.RMSE()
    model.score(test_it, metric)

    # Get predictions on test set
    preds = model.predict(test_it).asnumpy()
    print(preds)
    print(preds.shape)


if __name__ == "__main__":
    main()
