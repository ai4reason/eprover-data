#!/usr/bin/python

from atpy import *

BID="mizar40_5k"
LIMIT = "T60-G30000"
CORES = 60
VERSION = "VHSLCWh"


knn_methods = [('pv-mean', 'knn-512-pv-mean')] # Nickname for Mean method of watchlist selection and the directory containing the watchlist
for loop_num in range(4):
    for (km, wldir)in knn_methods:
        EARGS = {'eargs':"-s --training-examples=3 --free-numbers --record-proof-vector",# --watchlist-dir=01WLS/knn-512-pv-mean/mzr02",
                }
        PIDS=["mzr02-wlr"]
        wl_template = "01WLS/%s/mzr02" 
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
              "nthread": CORES,
              "num_round": 200,

           },
           "boosting": False,
           "efun": "EnigmaXgb",
           "hashing": 2**15
        }


        log.start("Starting experiment: enigmawatch with hashing", args)
       
        nick = km + '_'
        args['eargs']['nick'] = nick
        args['eargs']['wldir'] = wl_template % wldir
                
        model = "%s/%s/%s/%s/concat" % (VERSION, BID, PIDS[0], LIMIT)
        enigma.models.loop(model, PIDS, nick=nick, **args)
        n = 1 
        while n <= loop_num: 
            print(args['eargs']['wldir'])
            enigma.models.loop(model, PIDS, nick=nick+"loop0{}".format(n), **args)
            n += 1
        #enigma.models.loop(model, PIDS, nick=nick+"loop01", **args)
        #enigma.models.loop(model, PIDS, nick=nick+"loop02", **args)

#expres.dump.processed(BID, PIDS, results)
#expres.dump.solved(BID, PIDS, results, ref=PIDS[0])

