# Data for TABLEAUX'19 experiments #

This directory contains data for the paper submitted to TABLEAUX'19 conference.

## Benchmark problems ##

Benchmark problems can be downloaded here:

* [http://grid01.ciirc.cvut.cz/~mptp/7.13.01_4.181.1147/MPTP2/problems_small_consist.tar.gz][Mizar40]

The problem names from the subset of the 5000 problems used in the experiments
is provided in the file `problems.list`.

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

The scripts used to run the experiments are provided in the `scripts` directory.

