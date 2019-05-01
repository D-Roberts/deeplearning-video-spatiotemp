
# Plot losses
# plt = plot_losses(losses, 'w')
# # plt.show()
# plt.savefig('assets/losses_w')
# plt.close()

import numpy as np
from metric_util import rmse, plot_predictions

preds = np.loadtxt('/Users/denisaroberts/PycharmProjects/Lorenz/LorenzMap/assets/predictions/preds.txt')
labels = np.loadtxt('/Users/denisaroberts/PycharmProjects/Lorenz/LorenzMap/assets/predictions/labels.txt')

# predictions evaluations
rmse_test = rmse(preds, labels)
print('rmse test', rmse_test)
plt = plot_predictions(preds, labels)
# plt.savefig('assets/preds_w')
plt.show()
plt.close()
