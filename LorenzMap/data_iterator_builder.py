# TODO: data_iterator must come from another file
# x, y, z = generate_synthetic_lorenz(self.lorenz_steps)
# nTrain = len(x) - self.ntest
# train_x, test_x = x[:nTrain], x[nTrain:]
# train_y, test_y = y[:nTrain], y[nTrain:]
# train_z, test_z = z[:nTrain], z[nTrain:]
#
# if self.ts == 0:
#     train_data, test_data = train_x, test_x
# elif self.ts == 1:
#     train_data, test_data = train_y, test_y
# elif self.ts == 2:
#     train_data, test_data = train_z, test_z
#
#
# # Save for pred time
# np.savetxt('assets/predictions/test.txt', test_x)
#
# receptive_field = 2 ** self.dilation_depth
#
# data = np.append(np.zeros(receptive_field), train_data, axis=0)
#
# g = get_gluon_iterator(train_data, receptive_field=receptive_field, shuffle=True,
#                        batch_size=self.batch_size, last_batch='discard')



# TODO: both for train and predict;

# load predict input
receptive_field = 2 ** self.dilation_depth


# TODO: hard coded to be fixed
test_data = np.loadtxt('/Users/denisaroberts/PycharmProjects/Lorenz/LorenzMap/assets/predictions/test.txt')
g = get_gluon_iterator(test_data, receptive_field=receptive_field, shuffle=False,
                       batch_size=self.batch_size_predict, last_batch='discard')