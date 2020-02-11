# Data for IJCAR'20 experiments #

This directory contains data for the paper "ENIGMA Anonymous:
Symbol-Independent Inference Guiding Machine (system description)" submitted to
IJCAR'20 conference.

This repo contains the basic scripts.  Contact the authors for additional
information.


## Benchmark problems ##

Benchmark problems can be downloaded here:

http://grid01.ciirc.cvut.cz/~mptp/1147/MPTP2/problems_small_consist.tar.gz


## Baseline strategy ##

The baseline E strategy is in the file `mzr02`.


## Experiments with GBDT ##

1) The `eprover` with the GBDT support implemented by clause weight functions
`EnigmaLgb` and `EnigmaXgb` can be obtained here:

https://github.com/ai4reason/eprover/tree/devel

The exact commit used for the experiments is: 

https://github.com/ai4reason/eprover/tree/65490f2699b6aba7f8fbd72e847062d61d8e9ad4

2) Compilation requires XGBoost and LightGBM statically compiled with the
libraries `CONTRIB/lightgbm/liblightgbm.a`, `CONTRIB/xgboost/libxgboost.a`,
`CONTRIB/xgboost/librabit.a`, `CONTRIB/xgboost/libdmlc.a` placed therein.
Alternatively, you can adjust the Makefile for a dynamic compilation.
Statically pre-compiled binary is available in this repos in
`GDBT/eprover-gbdt`.

3) Scripts used for running the experiments are present in this repo in the
`GBDT` directory.  They use are Python repos `pyprove` and `enigmatic` obtainable here:

+ https://github.com/ai4reason/pyprove
+ https://github.com/ai4reason/enigmatic


## Experiments with GNN ##

1) The `eprover` with the GNN support implemented by clause weight function `EnigmaTf` can be obtained here:

https://github.com/ai4reason/eprover/tree/ETF

The exact commit used for the experiments is: 

https://github.com/ai4reason/eprover/tree/e034642a041505399c852c7e799b3e4047a1fc64

2) Compilation requires Tensorflow C API libraries placed in
`EXTERNAL/tensorflow/lib/libtensorflow.so` and
`EXTERNAL/tensorflow/lib/libtensorflow_framework.so`.

Some additional compilation information for the Tensorflow C API can be found
here: https://github.com/ai4reason/tensorflow-c-api-examples/

Dynamically compiled (as a static compilation is not easy) binary can be found
in this repo in `GNN/eprover-gnn`.

3) Scripts used for running the experiments are present in this repo in the
`GNN` directory.  Subdirectory `01-build` contains the script to build GNN
model, `02-convert` contains scripts to convert a network snapshot to a saved
version usable by E, and `03-evaluate` contains scripts to evaluate a model. 

