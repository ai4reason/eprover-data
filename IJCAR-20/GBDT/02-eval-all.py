#!/usr/bin/env python3

from pyprove import *

PIDS=open("eval.enigma").read().strip().split("\n")

experiment = {
   "bid"   : "mizar40/all",
   "pids"  : PIDS,
   "limit" : "T10",
   "cores" : 68,
   "eargs" : "-s --training-examples=3 --free-numbers"
}

log.start("Evaluating Enigma models", experiment)

experiment["results"] = expres.benchmarks.eval(**experiment)

expres.dump.processed(**experiment)
expres.dump.solved(**experiment)

