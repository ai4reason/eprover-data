#!/bin/bash

# force singlecore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ./PROVER/; timeout -k 1 15 ./eprover_sun24 --free-numbers -R -s --print-statistics --tstp-format -p --training-examples=3 --soft-cpu-limit=10 --cpu-limit=11 --definitional-cnf=24 --split-aggressive --simul-paramod --forward-context-sr --destructive-er-aggressive --destructive-er -tKBO6 -winvfreqrank -c1 -Ginvfreq -F1 --delete-bad-limit=150000000 -WSelectMaxLComplexAvoidPosPred  -H'(1*Torch(ConstPrio,paper-2-1))' $1

