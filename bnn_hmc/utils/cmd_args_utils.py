# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for command line arguments."""

import os
import sys
import tensorflow.compat.v2 as tf


def add_common_flags(parser):
  parser.add_argument("--tpu_ip", type=str, default=None,
                      help="Cloud TPU internal ip "
                           "(see `gcloud compute tpus list`)")
  parser.add_argument("--seed", type=int, default=0, help="Random seed")
  parser.add_argument("--temperature", type=float, default=1.,
                      help="Temperature of the posterior")
  parser.add_argument("--init_checkpoint", type=str, default=None,
                      help="Checkpoint to use for initialization of the chain")
  parser.add_argument("--tabulate_freq", type=int, default=40,
                      help="Frequency of tabulate table header prints")
  parser.add_argument("--dir", type=str, default=None, required=True,
                      help="Directory for checkpoints and tensorboard logs")
  parser.add_argument("--dataset_name", type=str, default="cifar10",
                      help="Name of the dataset")
  parser.add_argument("--subset_train_to", type=int, default=None,
                      help="Size of the subset of train data to use; "
                           "full dataset is used by default")
  parser.add_argument("--model_name", type=str, default="lenet",
                      help="Name of the dataset")
  parser.add_argument("--use_float64", dest="use_float64", action="store_true",
                      help="Use float64 precision (does not work on TPUs)")

  # Prior
  parser.add_argument("--prior_family", type=str, default="Gaussian",
                      choices=["Gaussian", "ExpFNormP", "Laplace", "StudentT",
                               "SumFilterLeNet", "EmpCovLeNet", "EmpCovMLP"],
                      help="Prior distribution family")
  parser.add_argument("--weight_decay", type=float, default=15.,
                      help="Weight decay, equivalent to setting prior std")
  parser.add_argument("--studentt_degrees_of_freedom", type=float, default=1.,
                      help="Degrees of freedom in StudentT family")
  parser.add_argument("--expfnormp_power", type=float, default=2.,
                      help="Power of the F-norm in ExpFNormp family")
  parser.add_argument("--sumfilterlenet_weight_decay", type=float, default=15.,
                      help="Weight decay for the Laplace prior over the sum of"
                           "the filter weights in the first layer of LeNet5"
                           "for the SumFilterLeNet prior family")
  parser.add_argument("--empcov_invcov_ckpt", type=str, default=None,
                      help="Path to the .npy checkpoint containing the inverse "
                           "of the covariance matrix "
                           "for the EmpCov family")
  parser.add_argument("--empcov_wd", type=float, default=None,
                      help="Weight decay for the EmpCov term"
                           "for the EmpCov family")

def add_sgd_flags(parser):

  parser.add_argument("--init_step_size", type=float, default=1.e-6,
                      help="Initial SGD step size")
  parser.add_argument("--num_epochs", type=int, default=300,
                      help="Total number of SGD epochs iterations")
  parser.add_argument("--batch_size", type=int, default=80, help="Batch size")
  parser.add_argument("--eval_freq", type=int, default=10,
                      help="Frequency of evaluation (epochs)")
  parser.add_argument("--save_freq", type=int, default=50,
                      help="Frequency of checkpointing (epochs)")
  parser.add_argument("--momentum_decay", type=float, default=0.9,
                      help="Momentum decay parameter for SGD")


def save_cmd(dirname, tf_writer):
  command = " ".join(sys.argv)
  with open(os.path.join(dirname, "command.sh"), "w") as f:
    f.write(command)
    f.write("\n")
  if tf_writer is not None:
    with tf_writer.as_default():
      tf.summary.text("command", command, step=0,
                      description="Command line arguments")
