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

import argparse
import time
from train_predict import train_predict_cw, train_predict_w


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
    parser.add_argument('--trajectory', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--Lorenzsteps', type=int, default=1500)
    parser.add_argument('--test_size', type=int, default=500)
    # TODO: add additional necessary args


    config = vars(parser.parse_args())

    start = time.time()

    if config['model'] == 'cw':
        train_predict_cw(ts=config['trajectory'], ntest=config['test_size'], Lorenznsteps=config['Lorenzsteps'],
                         batch_size=config['batch_size'], epochs=config['epochs'])

    elif config['model'] == 'w':
        train_predict_w(ts=config['trajectory'], ntest=config['test_size'], Lorenznsteps=config['Lorenzsteps'],
                         batch_size=config['batch_size'], epochs=config['epochs'])

    end = time.time()

    print("Train and predict took ", end - start, " seconds.")

if __name__ == '__main__':
    main()