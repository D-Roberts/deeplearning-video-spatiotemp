
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