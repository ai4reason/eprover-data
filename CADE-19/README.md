# Data for CADE'19 experiments #

This directory contains data for the paper "ENIGMA-NG: Efficient Neural and
Gradient-Boosted Inference Guidance for E"
(https://arxiv.org/abs/1903.03182) submitted to CADE'19 conference.

## Benchmark problems ##

Benchmark problems can be downloaded here:
https://github.com/JUrban/MPTP2078 We used the bushy versions.

## How to run experiments with LIBLINEAR ##

### Prerequisites ###

1. You need our version of eprover with LibLinear and XGBoost support.
   Additionally you need Enigma feature extractor `enigma-features`.
   Statically compiled versions are provided in the `bin` directory.  The
   `bin` directory must be in your `PATH` environment variable.  The source
   codes can be found at https://github.com/ai4reason/eprover/tree/devel.  

2. You also need `train` and `predict` programs from LIBLINEAR.  Binaries
   are provided in `bin` or can be downloaded from our eprover github
   repository above (https://github.com/ai4reason/eprover/tree/devel).  See
   the directory `CONTRIB/liblinear`.  Alternatively, you can download and
   compile official LIBLINEAR
   (https://www.csie.ntu.edu.tw/~cjlin/liblinear).

3. Finally you need ATPy Python package.  The version used for the
   experiments is in the `atpy` directory.  The latest version can be found
   at github (https://github.com/ai4reason/atpy).  Make sure you set the
   `PYTHONPATH` environment variable so that your Python finds the package
   (that is, put the directory containing `atpy` to `PYTHONPATH`).

### Run scripts ###

The scripts to run Enigma with LIBLINEAR are provided in the `scripts`
directory (`enigma-liblin.py` for the experiments without hashing and
`enigma-liblin-hashing.py` for the experiments with hashing).  Always run
the scripts directly from the `CADE-19` directory, that is, like
`./scripts/enigma-liblin.py`.  The `BID` variable in the scripts must be set
to the directory with the benchmarks which is looked up in the `CADE-19`
directory.  Alternatively, you can set `ATPY_BENCHMARKS` environment
variable to select a different location to search in.  The `eprover`
strategy used in the experiments can be found in `strats/mzr02`.

## How to run experiments with XGBoost ##

### Prerequisites ###

1. You need the prerequisites 1 and 3 from the above LIBLINEAR section.

2. Additionally you need the `XGBoost` Python package.  You can use, for
   example, `pip install --user xgboost` command.

### Run scripts ###

The scripts to run Enigma with XGBoost are provided in the `scripts`
directory (`enigma-xgboost.py` for the experiments without hashing and
`enigma-xgboost-hashing.py` for the experiments with hashing).  Always run
the scripts directly from the `CADE-19` directory, that is, like
`./scripts/enigma-xgboost.py`.  The `BID` variable in the scripts must be
set to as in the case of LIBLINEAR.  Additionally set the environment
variable `OMP_NUM_THREADS` to the value `1`.

## How to run experiments with LibTorch ##

### Prerequisites ###

The sub-folder `NeuralE` contains an executable of `E` extened to support neural guidance.
You will need to download the `libtorch` library
from `https://download.pytorch.org/libtorch/cu90/libtorch-shared-with-deps-latest.zip`
and copy all the relevant library files (the `*.so` files) to `NeuralE/CONTRIB/torcheval/libtorch/lib'.

### Running the scripts ###

The two bash scripts at the top level of `NeuralE`, namely  `run_bushy_neur_soft10s.sh` and
`run_bushy_mzr02neur_soft10s.sh` each expect a tptp problem as a single argument.
The first runs a strategy based solely on the learned model. The second combines the previously
discovered strategy `mzr02` with the learned model as described in the paper.

## How to train a new neural model ##

### Prerequisities ###

The scripts require Python 3. The sub-folder `NeuralTrain` contains
the list of required modules in `requirements.txt` and these modules
can be installed by calling `pip install -r requirements.txt`. For
faster parsing use PyPy 3 (https://pypy.org/download.html) and the
modules required by it are in `requirements-pypy.txt`. The script
works also without PyPy 3, but preprocessing is much slower and still
you need to install `lark-parser` by calling `pip install -r
requirements-pypy.txt`.

### Running the scripts ### 

The sub-folder `NeuralTrain` contains a bash script that can be run by
`bash learning.sh`. The newly generated model will be in the folder
`model` and to replace the original model you have to copy the content
of the folder to `NeuralE/PROVER/models/paper-2-1`.
