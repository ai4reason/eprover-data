#!/usr/bin/python

from atpy import *

BID="mizar40_5k" # The problem location
PIDS=["mzr02"]
EARGS = {'eargs':"-s --training-examples=3 --free-numbers"}
LIMIT = "T60-G30000"

args = {
        "bid": BID,
        "pids": PIDS,
        "limit": LIMIT,
        "cores": 58,
        "eargs": EARGS
}

results = expres.benchmarks.eval(**args)

expres.dump.solved(BID, PIDS, results)
expres.dump.processed(BID, PIDS, results)

