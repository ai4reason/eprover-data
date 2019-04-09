#!/usr/bin/python

from atpy import *

BID="bushy-cnf"
PIDS=["mzr02"]
EARGS = "-s --training-examples=3 --free-numbers" 
LIMIT = 10
CORES = 30
VERSION = "VHSLC"

eprover.result.STATUS_OK.remove("Satisfiable")
eprover.result.STATUS_OUT.append("Satisfiable")

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
   "boosting": True,
   "efun": "Enigma"
}

model = "%s/%s/%s/%ss/smartboost" % (VERSION, BID, PIDS[0], LIMIT)
enigma.models.loop(model, PIDS, **args)
#enigma.models.loop(model, PIDS, nick="loop01", **args)
#enigma.models.loop(model, PIDS, nick="loop02", **args)
#enigma.models.loop(model, PIDS, nick="loop03", **args)

expres.dump.processed(BID, PIDS, results)
expres.dump.solved(BID, PIDS, results, ref=PIDS[0])

