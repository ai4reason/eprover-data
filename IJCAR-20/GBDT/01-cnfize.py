#!/usr/bin/env python3

from pyprove import expres, log

experiment = {
   "bid"     : "mizar40/all",
   "cores"   : 68,
}

log.start("CNFize Benchmark(s)", experiment)

expres.benchmarks.cnfize(**experiment)

