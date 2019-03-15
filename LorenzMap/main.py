

import matplotlib
matplotlib.use('TkAgg')
from data_util import getDataLorenz
from data_util import get_mc_mt_gluon_iterator
from metric_util import plot_losses, plot_predictions
from train_predict import train_net_SGD_gluon_mc
from train_predict import predict
from metric_util import rmse


def main():
    x, y, z = getDataLorenz(1500)
    nTest = 500
    nTrain = len(x) - nTest
    train_x, test_x = x[:nTrain], x[nTrain:]
    train_y, test_y = y[:nTrain], y[nTrain:]
    train_z, test_z = z[:nTrain], z[nTrain:]

    ts = 0
    batch_size = 32
    losses, net = train_net_SGD_gluon_mc(ts, train_x, train_y, train_z, in_channels=3, receptive_field=16,
                                         batch_size=batch_size, epochs=1, lr=0.001, l2_reg=0.001)

    # Plot losses
    plt = plot_losses(losses, 'wvcnx')
    plt.show()
    plt.savefig('loss mc')

    # Make predictions on train set
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
    plt.show()


if __name__ == '__main__':
    main()