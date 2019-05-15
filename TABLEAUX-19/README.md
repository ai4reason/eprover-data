# Data for TABLEAUX'19 experiments #

This directory contains data for the paper submitted to TABLEAUX'19 conference.

## Benchmark problems ##

Benchmark problems can be downloaded here:
https://github.com/JUrban/MPTP2078 We used the bushy versions.

## How to run experiments with XGBoost ##

### Prerequisites ###

1. You need our version of eprover with LibLinear and XGBoost support.
   Additionally you need Enigma feature extractor `enigma-features`.
   Statically compiled versions are provided in the `bin` directory.  The
   `bin` directory must be in your `PATH` environment variable.  The source
   codes can be found at https://github.com/ai4reason/eprover/tree/devel.  

2. Finally you need ATPy Python package.  The version used for the
   experiments is in the `atpy` directory.  The latest version can be found
   at github (https://github.com/ai4reason/atpy).  Make sure you set the
   `PYTHONPATH` environment variable so that your Python finds the package
   (that is, put the directory containing `atpy` to `PYTHONPATH`).

3. Additionally you need the `XGBoost` Python package.  You can use, for
   example, `pip install --user xgboost` command.

### Run scripts ###

The scripts to run the experiments are provided in the `scripts`
directory 

TODO

