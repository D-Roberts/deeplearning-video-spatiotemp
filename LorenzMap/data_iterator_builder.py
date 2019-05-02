

class DIterators(object):
    def __init__(self, batch_size, dilation_depth, model, LorenzSteps, test_size, trajectory, for_train=True):
        self.batch_size = batch_size
        self.dilation_depth = dilation_depth
        self.ts = trajectory
        self.model = model
        self.stepcount = LorenzSteps
        self.ntest = test_size
        self.for_train = for_train
    # TODO: argparse class since all these args come from config

    def build_iterator(self, data):
        T = data.shape[0]
        Xx = nd.zeros((T - receptive_field, receptive_field))
        Xy = nd.zeros((T - receptive_field, receptive_field))
        Xz = nd.zeros((T - receptive_field, receptive_field))
        y = nd.zeros((T - receptive_field, data.shape[1]))

        for i in range(T - receptive_field):
            Xx[i, :] = data[i:i + receptive_field, 0]
            Xy[i, :] = data[i:i + receptive_field, 1]
            Xz[i, :] = data[i:i + receptive_field, 2]
            # labels
            for j in range(data.shape[1]):
                y[i, j] = data[i + receptive_field, j]

        X3 = nd.zeros((T - receptive_field, data.shape[1], receptive_field))
        for i in range(X3.shape[0]):
            X3[i, :, :] = nd.concatenate([Xx[i, :].reshape((1, -1)), Xy[i, :].reshape((1, -1)), Xz[i, :].reshape((1, -1))])

        if self.model == 'cw':
            dataset = gluon.data.ArrayDataset(X3, y[:, self.ts])

        else:
            # TODO: the X should be refactored for easy selection
            if self.ts == 0:
                dataset = gluon.data.ArrayDataset(Xx, y[:, self.ts])

            elif self.ts == 1:
                dataset = gluon.data.ArrayDataset(Xy, y[:, self.ts])

            else:
                dataset = gluon.data.ArrayDataset(Xz, y[:, self.ts])

        if self.for_train:
            diter = gluon.data.DataLoader(dataset, self.batch_size, shuffle=True, last_batch='discard')
        else:
            diter = gluon.data.DataLoader(dataset, 1, shuffle=False, last_batch='discard')
        return diter

    def predict_iterator(self, predict_input):
        return self.build_iterator(predict_input)

    def train_iterator(self):
        data = generate_synthetic_lorenz(self.stepcount)
        receptive_field = self.dilation_depth ** 2
        self.nTrain = data.shape[0] - self.ntest
        self.train_data, self.test_data = data[:self.nTrain, :], data[self.nTrain:, :]
        self.train_data = np.append(np.zeros(receptive_field), self.train_data, axis=0)

        return self.build_iterator((self.train_data))



