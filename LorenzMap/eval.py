
import os

import numpy as np

from utils import rmse, plot_predictions, plot_losses

class Evaluate(object):
    """

    Load and plot training loss.

    Load predictions and calculate rmse.
    Plot predictions vs ground truth for the model run.
    """
    def __init__(self, options):
        self._options = options

    def evaluate_model(self):

        # train losses
        if self._options.epochs > 1:
            loss_save = np.loadtxt(os.path.join(self._options.assets_dir, 'losses.txt'))
            plt = plot_losses(loss_save, 'train loss')
            plt.show()
            plt.savefig(os.path.join(self._options.assets_dir, 'train_loss'))
            plt.close()

        # predictions
        preds = np.loadtxt(os.path.join(self._options.assets_dir, 'preds.txt'))
        labels = np.loadtxt(os.path.join(self._options.assets_dir, 'labels.txt'))

        rmse_test = rmse(preds, labels)
        print('RMSE test set', rmse_test)
        plt = plot_predictions(preds, labels)
        plt.savefig(os.path.join(self._options.assets_dir, 'preds_plot'))
        plt.show()
        plt.close()

    def __call__(self):
        self.evaluate_model()
