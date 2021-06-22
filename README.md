# Dangers of Bayesian Model Averaging under Covariate Shift

This repository contains the code to reproduce the experiments 
in the paper
_Dangers of Bayesian Model Averaging under Covariate Shift_
by Pavel Izmailov, Patrick Nicholson, Sanae Lotfi and Andrew Gordon Wilson.

The code is forked from the Google Research [BNN HMC repo](https://github.com/google-research/google-research/tree/master/bnn_hmc).

## Introduction

Approximate Bayesian inference for neural networks is considered a robust alternative to standard training, often providing good performance on out-of-distribution data.
However, it was recently [shown](https://arxiv.org/abs/2104.14421) that Bayesian neural networks (BNNs) with high fidelity inference through Hamiltonian Monte Carlo (HMC) provide shockingly poor performance under covariate shift.
For example, in the left panel of the figure below we see that a ResNet-20 BNN approximated with HMC underperforms a maximum a-posteriori (MAP) solution by $25\%$ on the _pixelate_-corrupted CIFAR-10 test set. 
This result is particularly surprising given that on the in-distribution test data, the BNN outperforms the MAP solution by over 5%.
In this work, we seek to understand, further demonstrate, and help remedy this concerning behaviour. 

<p align="center">
<table>
  <tr>
    <th><img src="https://user-images.githubusercontent.com/14368801/122979309-65529280-d365-11eb-97cc-a7106cb89c86.png" height=180></th>
    <th><img src="https://user-images.githubusercontent.com/14368801/122979306-64b9fc00-d365-11eb-86e5-90637403bcdb.png" height=180></th>
    <th><img src="https://user-images.githubusercontent.com/14368801/122979307-64b9fc00-d365-11eb-88d0-d9ef88a16c73.png" height=180></th>
  </tr>
  <tr>
    <th>ResNet20, CIFAR-10</th>
    <th>BNN weight sample</th>
    <th>MAP solution</th>
  </tr>
</table>
</p>
<!--
Intuitively, we find that Bayesian model averaging (BMA) can be problematic under covariate shift as follows.
Due to dependencies in the features of the train data distribution, model parameters corresponding to these dependencies do not affect the predictions on the train data. For example, parameters connected to dead pixels, i.e. pixels with intensity zero across all train images, do not affect predictions. 
For these parameters, the posterior coincides with the prior.
The MAP solution sets the values of these parameters to zero, due to regularization from the prior that penalizes the parameter norm, while the BMA samples these weights from the prior.
At test time, the model is applied to a different data distribution, where the features do not have the same dependence, and the parameters that did not affect the predictions on train can negatively affect predictions on test.

In \autoref{fig:intro_figure}(b, c) we visualize the weights in the first layer of a fully-connected network for a sample from the BNN posterior and the MAP solution on the MNIST dataset.
The MAP solution weights are highly structured, while the BNN sample appears extremely noisy, similar to a draw from the Gaussian prior.
In particular the weights corresponding to \textit{dead pixels} (i.e. pixel positions that are black for all the MNIST images) near the boundary of the input image are set near zero (shown in white) by the MAP solution, but sampled randomly by the BNN.
If at test time the data is corrupted, e.g. by Gaussian noise, and the pixels near the boundary of the image are activated,
the MAP solution will ignore these pixels, while the predictions of the BNN will be significantly affected. 
-->



## Requirements

We use provide a `requirements.txt` file that can be used to create a conda
environment to run the code in this repo:
```bash
conda create --name <env> --file requirements.txt
```

Example set-up using `pip`:
```bash
pip install tensorflow

pip install --upgrade pip
pip install --upgrade jax jaxlib==0.1.65+cuda112 -f \
https://storage.googleapis.com/jax-releases/jax_releases.html

pip install git+https://github.com/deepmind/dm-haiku
pip install tensorflow_datasets
pip install tabulate
pip install optax
```
Please see the [_JAX repo_](https://github.com/google/jax) for the latest
instructions on how to install JAX on your hardware.

## File Structure

ToDo

```
.
+-- core/
|   +-- hmc.py (The Hamiltonian Monte Carlo algorithm)
|   +-- sgmcmc.py (SGMCMC methods as optax optimizers)
|   +-- vi.py (Mean field variational inference)
+-- utils/ (Utility functions used by the training scripts)
|   +-- train_utils.py (The training epochs and update rules)
|   +-- models.py (Models used in the experiments)
|   +-- losses.py (Prior and likelihood functions)
|   +-- data_utils.py (Loading and pre-processing the data)
|   +-- optim_utils.py (Optimizers and learning rate schedules)
|   +-- ensemble_utils.py (Implementation of ensembling of predictions)
|   +-- metrics.py (Metrics used in evaluation)
|   +-- cmd_args_utils.py (Common command line arguments)
|   +-- script_utils.py (Common functionality of the training scripts)
|   +-- checkpoint_utils.py (Saving and loading checkpoints)
|   +-- logging_utils.py (Utilities for logging printing the results)
|   +-- precision_utils.py (Controlling the numerical precision)
|   +-- tree_utils.py (Common operations on pytree objects)
+-- run_hmc.py (HMC training script)
+-- run_sgd.py (SGD training script)
+-- run_sgmcmc.py (SGMCMC training script)
+-- run_vi.py (MFVI training script)
+-- make_posterior_surface_plot.py (script to visualize posterior density)
```

## Training Scripts

ToDo

Common command line arguments:

* `seed` &mdash; random seed
* `dir` &mdash; training directory for saving the checkpoints and 
tensorboard logs
* `dataset_name` &mdash; name of the dataset, e.g. `cifar10`, `cifar100`, 
  `imdb`; 
  for the UCI datasets, the name is specified as 
  `<UCI dataset name>_<random seed>`, e.g. `yacht_2`, where the seed determines
  the train-test split
* `subset_train_to` &mdash; number of datapoints to use from the dataset;
  by default, the full dataset is used
* `model_name` &mdash; name of the neural network architecture, e.g. `lenet`, 
  `resnet20_frn_swish`, `cnn_lstm`, `mlp_regression_small` 
* `weight_decay` &mdash; weight decay; for Bayesian methods, weight decay
determines the prior variance (`prior_var = 1 / weight_decay`)
* `temperature` &mdash; posterior temperature (default: `1`)
* `init_checkpoint` &mdash; path to the checkpoint to use for initialization
  (optional)
* `tabulate_freq` &mdash; frequency of tabulate table header logging
* `use_float64` &mdash; use float64 precision (does not work on TPUs and some
  GPUs); by default, we use `float32` precision

### Running HMC

To run HMC, you can use the `run_hmc.py` training script. Arguments:

* `step_size` &mdash; HMC step size
* `trajectory_len` &mdash; HMC trajectory length
* `num_iterations` &mdash; Total number of HMC iterations
* `max_num_leapfrog_steps` &mdash; Maximum number of leapfrog steps allowed; 
  meant as a sanity check and should be greater than 
  `trajectory_len / step_size`
* `num_burn_in_iterations` &mdash; Number of burn-in iterations (default: `0`)

#### Examples

ToDo

```bash
ToDo
```
We ran these commands on a machine with 8 NVIDIA Tesla V-100 GPUs.

MLP on a subset of 160 datapoints from MNIST:
```bash
python3 run_hmc.py --seed=0 --weight_decay=1. --temperature=1. \
  --dir=runs/hmc/mnist_subset160 --dataset_name=mnist \
  --model_name=mlp_classification --step_size=3.e-5 --trajectory_len=1.5 \ 
  --num_iterations=100 --max_num_leapfrog_steps=50000 \
  --num_burn_in_iterations=10 --subset_train_to=160
```
This script can be ran on a single GPU.

**Note**: we run HMC on CIFAR-10 on TPU pod with 512 TPU devices with a
modified version of the code that we will release soon.

### Running SGD

To run SGD, you can use the `run_sgd.py` training script. Arguments:

* `init_step_size` &mdash; Initial SGD step size; we use a cosine schedule
* `num_epochs` &mdash; total number of SGD epochs iterations
* `batch_size` &mdash; batch size
* `eval_freq` &mdash; frequency of evaluation (epochs)
* `save_freq` &mdash; frequency of checkpointing (epochs)
* `momentum_decay` &mdash; momentum decay parameter for SGD

#### Examples

ToDo
```bash
ToDo
```

To train a deep ensemble, we simply train multiple copies of SGD with different
random seeds.

## Results

ToDo

![combined_resolution png-1](https://user-images.githubusercontent.com/14368801/122981650-fd517b80-d367-11eb-9876-52a26cbd0200.png)
