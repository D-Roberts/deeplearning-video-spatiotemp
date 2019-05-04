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

import sys
import argparse
from pprint import pprint


class ArgParser(object):
    """

    Parse train or predict related parameters from command line args.
    """

    def __init__(self):
        self._arg_parser = argparse.ArgumentParser()

    def parse_args(self, for_train):
        """

        :param for_train: for train or predict.
        :return: the dict of parsed args.
        """
        self._set_global_arg()
        if for_train:
            self._set_train_arg()
        else:
            self._set_predict_arg()
        self._set_model_parameter_arg()
        options, args = self._arg_parser.parse_known_args()
        pprint(options, width=1)
        sys.stdout.flush()
        return for_train, options

    def _set_global_arg(self):
        """

        Interface implementation arguments.
        """

        self._arg_parser.add_argument('--train', help='train mode or not', default=True)
        self._arg_parser.add_argument('--trajectory', help='which trajectory to build', default=0)
        self._arg_parser.add_argument('--model', help='conditional or unconditional', default='cw')
        self._arg_parser.add_argument('--in_channels', type=int, default=3)
        self._arg_parser.add_argument('--lorenz_steps', type=int, default=1500, help='Synthetic data generation')
        self._arg_parser.add_argument('--predict', type=bool, default=True, help='If to do pred at the same time')
        self._arg_parser.add_argument('--test_size', type=int, default=500)

    def _set_train_arg(self):
        """

        Model training related arguments.
        """

        self._arg_parser.add_argument('--plot_losses', type=bool, default=True)
        self._arg_parser.add_argument('--check_path', type=str, default='assets/best_perf_model')
        self._arg_parser.add_argument('--predict_input_path', type=str, default='assets/predictions/test.txt')


    def _set_predict_arg(self):
        """

        Prediction related arguments.
        """

        self._arg_parser.add_argument('--check_path', type=str, default='assets/best_perf_model')
        self._arg_parser.add_argument('--predict_input_path', type=str, default='assets/predictions/test.txt')
        self._arg_parser.add_argument('--evaluation', type=bool, default=True)
        self._arg_parser.add_argument('--batch_size_predict', type=int, default=1)

    def _set_model_parameter_arg(self):
        """

        Model parameters for training and prediction.
        """

        default_gpu_num = 0
        default_epochs = 200
        default_batch_size = 32
        default_dilation_depth = 4
        default_learning_rate = 0.001
        default_l2_regularization = 0.001
        default_test_size = 500

        # TODO: setting the context issue.
        self._arg_parser.add_argument('--batch_size', type=int, default=default_batch_size)
        self._arg_parser.add_argument('--epochs', type=int, default=default_epochs)
        self._arg_parser.add_argument('--dilation_depth', type=int, default=default_dilation_depth)
        self._arg_parser.add_argument('--learning_rate', type=float, default=default_learning_rate)
        self._arg_parser.add_argument('--l2_regularization', type=float, default=default_l2_regularization)



