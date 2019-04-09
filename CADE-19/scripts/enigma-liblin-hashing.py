#!/usr/bin/python

from atpy import *

BID="bushy-cnf"
PIDS=["mzr02"]
EARGS = "-s --training-examples=3 --free-numbers"
LIMIT = 10
CORES = 60
VERSION = "VHSLCh"

results = {}
args = {
   "results": results,
   "update": True,
   "bid": BID,
   "limit": LIMIT,
   "cores": CORES,
   "version": VERSION,
   "eargs": EARGS,
   "xgb": False,
   #"xgb_params": {
   #   "lambda": 0,
   #   "alpha": 0,
   #   "max_depth": 9,
   #   "nthreads": CORES,
   #   #"tree_method": "exact",
   #   "num_round": 200
   #},
   "boosting": True,
   "efun": "Enigma",
   "hashing": 2**14
}


for h in [32, 16, 8, 4 , 2, 1]:
	results = {}
	args["results"] = results
	args["hashing"] = h*1024
	PIDS = ["mzr02"]
	
	log.start("Starting experiment: enigma with hashing", args)
	
	model = "%s/%s/%s/%ss/concat-hash%sk" % (VERSION, BID, PIDS[0], LIMIT, h)
	enigma.models.loop(model, PIDS, **args)
	#enigma.models.loop(model, PIDS, nick="loop01", **args)
	#enigma.models.loop(model, PIDS, nick="loop02", **args)
	#enigma.models.loop(model, PIDS, nick="loop03", **args)
	
	expres.dump.processed(BID, PIDS, results)
	expres.dump.solved(BID, PIDS, results, ref=PIDS[0])

