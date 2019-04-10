#!/bin/bash

# force singlecore
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd ./PROVER/; timeout -k 1 15 ./eprover_sun24 --free-numbers -R -s --print-statistics --tstp-format -p --training-examples=3 --soft-cpu-limit=10 --cpu-limit=11 --definitional-cnf=24 --split-aggressive --simul-paramod --forward-context-sr --destructive-er-aggressive --destructive-er -tKBO6 -winvfreqrank -c1 -Ginvfreq -F1 --delete-bad-limit=150000000 -WSelectMaxLComplexAvoidPosPred  -H'(1*ConjectureTermPrefixWeight(DeferSOS,1,3,0.1,5,0,0.1,1,4),1*ConjectureTermPrefixWeight(DeferSOS,1,3,0.5,100,0,0.2,0.2,4),1*Refinedweight(PreferWatchlist,4,300,4,4,0.7),1*RelevanceLevelWeight2(PreferProcessed,0,1,2,1,1,1,200,200,2.5,9999.9,9999.9),1*StaggeredWeight(DeferSOS,1),1*SymbolTypeweight(DeferSOS,18,7,-2,5,9999.9,2,1.5),2*Clauseweight(PreferWatchlist,20,9999,4),2*ConjectureSymbolWeight(DeferSOS,9999,20,50,-1,50,3,3,0.5),2*StaggeredWeight(DeferSOS,2),12*Torch(ConstPrio,paper-2-1))' $1
