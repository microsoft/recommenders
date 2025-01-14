<!--
Copyright (c) Recommenders contributors.
Licensed under the MIT License.
-->

# Recommender Utilities

This package contains functions to simplify common tasks used when developing and evaluating recommender systems. A short description of the submodules is provided below. For more details about what functions are available and how to use them, please review the doc-strings provided with the code or the [online documentation](https://readthedocs.org/projects/microsoft-recommenders/).

# Installation

## Pre-requisites
Some dependencies require compilation during pip installation. On Linux this can be supported by adding build-essential dependencies:
```bash
sudo apt-get install -y build-essential libpython<version>
``` 
where `<version>` should be the Python version (e.g. `3.8`).

On Windows you will need [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

For more details about the software requirements that must be pre-installed on each supported platform, see the [setup guide](https://github.com/microsoft/recommenders/blob/main/SETUP.md).   

## Basic installation

To install core utilities, CPU-based algorithms, and dependencies
```bash
pip install --upgrade pip setuptools
pip install recommenders
```

## Optional Dependencies

By default `recommenders` does not install all dependencies used throughout the code and the notebook examples in this repo. Instead we require a bare minimum set of dependencies needed to execute functionality in the `recommenders` package (excluding Spark, GPU and Jupyter functionality). We also allow the user to specify which groups of dependencies are needed at installation time (or later if updating the pip installation). The following groups are provided:

- examples: dependencies related to Jupyter needed to run [example notebooks](https://github.com/microsoft/recommenders/tree/main/examples)
- gpu: dependencies to enable GPU functionality (PyTorch & TensorFlow)
- spark: dependencies to enable Apache Spark functionality used in dataset, splitting, evaluation and certain algorithms
- dev: dependencies such as `black` and `pytest` required only for development or testing
- all: all of the above dependencies
- experimental: current experimental dependencies that are being evaluated (e.g. libraries that require advanced build requirements or might conflict with libraries from other options)
- nni: dependencies for NNI tuning framework.

Note that, currently, xLearn and Vowpal Wabbit are in the experimental group.

These groups can be installed alone or in combination:
```bash
# install recommenders with core requirements and support for CPU-based recommender algorithms and notebooks
pip install recommenders[examples]

# add support for running example notebooks and GPU functionality
pip install recommenders[examples,gpu]
```

## GPU Support

You will need CUDA Toolkit v11.2 and CuDNN v8.1 to enable both Tensorflow and PyTorch to use the GPU. For example, if you are using a conda environment, this can be installed with
```bash
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1
```
For a virtual environment, you may use a [docker container by Nvidia](../SETUP.md#using-a-virtual-environment). 

For manual installation of the necessary requirements see [TensorFlow](https://www.tensorflow.org/install/gpu#software_requirements) and [PyTorch](https://pytorch.org/get-started/locally/) installation pages.

When installing with GPU support you will need to point to the PyTorch index to ensure you are downloading a version of PyTorch compiled with CUDA support. This can be done using the --find-links or -f option below.

`pip install recommenders[gpu] -f https://download.pytorch.org/whl/cu111/torch_stable.html`

## Experimental dependencies

We are currently evaluating inclusion of the following dependencies:

 - vowpalwabbit: current examples show how to use vowpal wabbit after it has been installed on the command line; using the [PyPI package](https://pypi.org/project/vowpalwabbit/) with the scikit-learn interface will facilitate easier integration into python environments
 - xlearn: on some platforms, xLearn requires pre-installation of cmake.

## Other dependencies

Some dependencies are not available via the recommenders PyPI package, but can be installed with the [requirements-external.txt](./requirements-external.txt) file:

```
pip install -r requirements-external.txt
```

## NNI dependencies

For NNI a more recent version can be installed but is untested.


## Installing the utilities from a local copy

In case you want to use a version of the source code that is not published on PyPI, one alternative is to install from a clone of the source code on your machine. To this end, 
a [setup.py](../setup.py) file is provided in order to simplify the installation of the utilities in this repo from the main directory.

This still requires an environment to be installed as described in the [setup guide](../SETUP.md). Once the necessary dependencies are installed, you can use the following command to install `recommenders` as a python package.

    pip install -e .

It is also possible to install directly from GitHub. Or from a specific branch as well.

    pip install -e git+https://github.com/microsoft/recommenders/#egg=pkg
    pip install -e git+https://github.com/microsoft/recommenders/@staging#egg=pkg

**NOTE** - The pip installation does not install all of the pre-requisites; it is assumed that the environment has already been set up according to the [setup guide](../SETUP.md), for the utilities to be used.


# Contents

## [Datasets](datasets)

Datasets module includes helper functions for pulling different datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

### Data Loading

There are dataloaders for several datasets. For example, the movielens module will allow you to load a dataframe in pandas or spark formats from the MovieLens dataset, with sizes of 100k, 1M, 10M, or 20M to test algorithms and evaluate performance benchmarks.

```python
df = movielens.load_pandas_df(size="100k")
```

### Splitting Techniques

Currently three methods are available for splitting datasets. All of them support splitting by user or item and filtering out minimal samples (for instance users that have not rated enough items, or items that have not been rated by enough users).

- Random: this is the basic approach where entries are randomly assigned to each group based on the ratio desired
- Chronological: this uses provided timestamps to order the data and selects a cut-off time that will split the desired ratio of data to train before that time and test after that time
- Stratified: this is similar to random sampling, but the splits are stratified, for example if the datasets are split by user, the splitting approach will attempt to maintain the same ratio of items used in both training and test splits. The converse is true if splitting by item.

## [Evaluation](evaluation)

The evaluation submodule includes functionality for calculating common recommendation metrics directly in Python or in a Spark environment using PySpark.

Currently available metrics include:

- Root Mean Squared Error
- Mean Absolute Error
- R<sup>2</sup>
- Explained Variance
- Precision at K
- Recall at K
- Normalized Discounted Cumulative Gain at K
- Mean Average Precision at K
- Area Under Curve
- Logistic Loss

## [Models](models)

The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new recommender system approaches. A description of all the algorithms can be found on [this table](../README.md#algorithms). The following is a list of the algorithm utilities:

* Cornac
* DeepRec
  *  Convolutional Sequence Embedding Recommendation (CASER)
  *  Deep Knowledge-Aware Network (DKN)
  *  Extreme Deep Factorization Machine (xDeepFM)
  *  GRU
  *  LightGCN
  *  Next Item Recommendation (NextItNet)
  *  Short-term and Long-term Preference Integrated Recommender (SLi-Rec)
  *  Multi-Interest-Aware Sequential User Modeling (SUM)
* FastAI
* GeoIMC
* LightFM
* LightGBM
* NCF
* NewsRec
  * Neural Recommendation with Long- and Short-term User Representations (LSTUR)
  * Neural Recommendation with Attentive Multi-View Learning (NAML)
  * Neural Recommendation with Personalized Attention (NPA)
  * Neural Recommendation with Multi-Head Self-Attention (NRMS)
* Restricted Boltzmann Machines (RBM)
* Riemannian Low-rank Matrix Completion (RLRMC)
* Simple Algorithm for Recommendation (SAR)
* Self-Attentive Sequential Recommendation (SASRec)
* Sequential Recommendation Via Personalized Transformer (SSEPT)
* Surprise
* Term Frequency - Inverse Document Frequency (TF-IDF)
* Variational Autoencoders (VAE)
  * Multinomial
  * Standard
* Vowpal Wabbit (VW)
* Wide and Deep
* xLearn
  * Factorization Machine (FM)
  * Field-Aware FM (FFM)

## [Tuning](tuning)

This submodule contains utilities for performing hyperparameter tuning.

## [Utils](utils)

This submodule contains high-level utilities for defining constants used in most algorithms as well as helper functions for managing aspects of different frameworks: GPU, Spark, Jupyter notebook.
