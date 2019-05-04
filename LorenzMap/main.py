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


import time
import argparse
from model_train import Train
from model_predict import Predict
from arg_parser import ArgParser


def main():
    """Run train and predict for the various Lorenz map prediction models with user
    provided arguments. Assets are saved in the 'assets' folder in the project directory.
    Assets saved are best model, losses plot and predictions vs ground truth plot.

    model: can be Conditional Wavenet (cw), Unconditional Wavenet (w),
    Conditional LSTM (clstm), Unconditional LSTM (lstm).

    trajectory: to predict x (ts=0), y(ts=1), or z(ts=2) Lorenz trajectories.

    epochs: default for wavenet =100, default for lstm =30.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cw')
    parser.add_argument('--trajectory', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lorenz_steps', type=int, default=1500)
    parser.add_argument('--test_size', type=int, default=500)
    parser.add_argument('--dilation_depth', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--l2_regularization', type=float, default=0.001)
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--checkp_path', type=str, default='assets/best_perf_model')
    parser.add_argument('--predict', type=bool, default=True)
    parser.add_argument('--predict_input_path', type=str, default='assets/predictions/test.txt')
    parser.add_argument('--evaluation', type=bool, default=True)
    parser.add_argument('--plot_losses', type=bool, default=True)

    options = ArgParser.parse_args(for_train=True)
    config = parser.parse_args()
    trainer = Train(config)
    predictor = Predict(config)

    start = time.time()
    if config.train:
        trainer.train()
    if config.predict:
        predictor.predict()

    end = time.time()
    print("Process took ", end - start, " seconds.")

if __name__ == '__main__':
    main()



    # TODO: argparse cofnig object
    # TODO: use argparse object in main to get config and pass config to where necessary in other modules

    # TODO: get gpu ready code.

    # TODO: test that all works at this point and push to git



    # --------------------

    # TODO: consider if RMSE applies or not.
    # TODO: evaluate if she reported tests on the param selection she says or on
    # the setting she had in the code and used in my own tuning.

    # TODO: rerun performances. Notice better with 200 epochs on z unconditional.
    # TODO: recheck that the correct ts are predicted; seems weirdly the same performance and plot.
    # seems that cw requires now 400 epochs or more to get a score higher than w. weird. test the
    # other params for lorenz; and better perform for y traj. (0.002 or so)

    # performance on z is terrible : 0.14