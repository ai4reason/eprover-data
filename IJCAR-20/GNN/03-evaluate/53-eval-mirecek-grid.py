#!/usr/bin/env python3

from pyprove import *

PIDS=open("eval").read().strip().split("\n")
#PIDS=["Enigma+mizar40-all-T5+greedy5-G_E___207_C01_F1_SE_CS_SP_PI_S0Y-VHSLCXPh+lgb-d16-l600-e0.2+loop05+coop"]
#PIDS=["mzr02-mirecek","mzr02"]
#PIDS=["mzr02-mirecek","mzr02-mirecek-solo"]

experiment = {
   "bid"   : "mizar40/all/tenth",
   "pids"  : PIDS,
   #"limit" : "G5000-T60",
   "limit" : "T10",
   "cores" : 60,
   "eargs" : "-s --training-examples=3 --free-numbers",
   "ebinary": "/home/yan/repos/eprover-ETF/PROVER/eprover"
}

log.start("Evaluating Enigma models", experiment)

experiment["results"] = expres.benchmarks.eval(**experiment)

expres.dump.processed(**experiment)
expres.dump.solved(**experiment)

