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


from model_train import Train
from model_predict import Predict
from data_generation import LorenzMapData
from data_iterator_builder import DIterators
from arg_parser import ArgParser
from eval import Evaluate

def main():
    """Run train and predict for the various Lorenz map prediction models with user
    provided arguments. Assets are saved in the 'assets' folder in the project directory.
    Assets saved are best model, losses plot and predictions vs ground truth plot.

    model: can be Conditional Wavenet (cw), Unconditional Wavenet (w),
    Conditional LSTM (clstm), Unconditional LSTM (lstm).

    trajectory: to predict x (ts=0), y(ts=1), or z(ts=2) Lorenz trajectories.

    epochs: default for wavenet =100, default for lstm =30.
    """


    argparser = ArgParser()
    options = argparser.parse_args()
    data_generator = LorenzMapData(options)
    train_data, test_data = data_generator.generate_train_test_sets()

    # Train
    trainer = Train(options)
    train_iter = DIterators(options).build_iterator(train_data, for_train=True)
    trainer.train(train_iter)

    # Predict on test set and evaluate
    predictor = Predict(options)
    predict_iter = DIterators(options).build_iterator(test_data, for_train=False)
    predictor.predict(predict_iter)

    # Evaluate performance on test set
    evaluator = Evaluate(options)
    evaluator()


if __name__ == '__main__':
    main()


    # TODO: separate evaluate from predict; predict should only save

    # TODO: format Pep 8 (look for arg alignment in mxnet code)


    # TODO: test that all works at this point and push to git

    #TODO; sufficient to inquire

    # --------------------

    # TODO: evaluate if she reported tests on the param selection she says or on
    # the setting she had in the code and used in my own tuning.

    # TODO: rerun performances. Notice better with 200 epochs on z unconditional.
    # TODO: recheck that the correct ts are predicted; seems weirdly the same performance and plot.
    # seems that cw requires now 400 epochs or more to get a score higher than w. weird. test the
    # other params for lorenz; and better perform for y traj. (0.002 or so)

    # performance on z is terrible : 0.14