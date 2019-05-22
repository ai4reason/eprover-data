#!/usr/bin/python

from atpy import *

BID="mizar40_5k"
PIDS=["mzr02"]
EARGS = {'eargs':"-s --training-examples=3 --free-numbers"}
#LIMIT = 10
LIMIT = "T60-G30000"
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
   "xgb": True,
   "xgb_params": {
      "lambda": 0,
      "alpha": 0,
      "max_depth": 9,
      "nthreads": CORES,
      #"tree_method": "exact",
      "num_round": 200
   },
   "boosting": False,
   "efun": "EnigmaXgb",
   "hashing": 2**15
}

log.start("Starting experiment: enigma with hashing", args)

model = "%s/%s/%s/%ss/concat-hash/" % (VERSION, BID, PIDS[0], LIMIT)
nick = ''# 'knn-512-pv-mean-npv_'
enigma.models.loop(model, PIDS, nick=nick, **args)
enigma.models.loop(model, PIDS, nick=nick+"loop01", **args)
enigma.models.loop(model, PIDS, nick=nick+"loop02", **args)
enigma.models.loop(model, PIDS, nick=nick+"loop03", **args)
enigma.models.loop(model, PIDS, nick=nick+"loop04", **args)
enigma.models.loop(model, PIDS, nick=nick+"loop05", **args)

expres.dump.processed(BID, PIDS, results)
expres.dump.solved(BID, PIDS, results, ref=PIDS[0])

