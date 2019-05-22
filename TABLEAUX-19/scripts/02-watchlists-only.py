#!/usr/bin/python

from atpy import *

BID="mizar40_5k"
LIMIT = "T60-G30000" 
EARGS = {'eargs':"-s --training-examples=3 --free-numbers --record-proof-vector", 
}

knn_methods = [('pv-mean', 'knn-512-pv-mean')] # Nickname for Mean method of watchlist selection and the directory containing the watchlist
wl_template = "--watchlist-dir=01WLS/%s"

args = {
        "bid": BID,
        "limit": LIMIT,
        "cores": 58,
        "eargs": EARGS
}

results = {}
for (km, wldir)in knn_methods:
    PIDS=["mzr02"]
    for PID in PIDS:
            args['eargs']['nick'] = km + '_'
            args['eargs']['wl'] = wl_template % wldir
            args['eargs']['pid'] = PID.replace("-wlr","")
            args["pids"] = [PID]
            results.update(expres.benchmarks.eval(**args))

#expres.dump.solved(BID, result_PIDS, results)
#expres.dump.processed(BID, result_PIDS, results)

