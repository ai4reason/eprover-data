#!/usr/bin/env python3

from pyprove import log, expres
from enigmatic import models, learn, protos

#learning = {
#   'learning_rate': 0.29, 
#   'num_leaves': 999, 
#   'n_estimators': 150, 
#   'num_round': 150, 
#   'reg_lambda': 0, 
#   'random_state': 42, 
#   'objective': 'binary', 
#   'min_child_samples': 40, 
#   'max_depth': 120,
#   'n_jobs': 68
#}


new = []

for DEPTH in [30,40,50,60,80]:
#for DEPTH in [40,50,60,80,100,150]:
#for DEPTH in [50,100,150]:
#for DEPTH in [10,20,30,40]:
   #for LEAVES in [600,900,1200,1500]:
   for LEAVES in [1800]: #[900,1200,1500]:
      #for ETA in [0.2,0.25]:
      for ETA in [0.15]:
      
         learning = {
            'learning_rate': ETA, 
            'num_leaves': LEAVES, 
            'num_round': 150, 
            'reg_lambda': 0, 
            'random_state': 42, 
            'objective': 'binary', 
            'min_child_samples': 40, 
            'max_depth': DEPTH,
         }
         
         settings = {
            "bid"     : "mizar40/all",
            "pids"    : [
                'mzr02',
                "Enigma+mizar40-all-T10+mzr02-VHSLCAXPh+xgb-d12-e0.2+solo",
                "Enigma+mizar40-all-T10+mzr02-VHSLCAXPh+xgb-d12-e0.2+coop",
                "Enigma+mizar40-all-T10+mzr02-VHSLCAXPh+xgb-d12-e0.2-loop01+solo",
                "Enigma+mizar40-all-T10+mzr02-VHSLCAXPh+xgb-d12-e0.2-loop01+coop"
            ],
            "ref"     : "loop02",
            "limit"   : "T10",
            "gzip"    : False,
            "cores"   : 60,
            "version" : "VHSLCAXPh",
            #"eargs"   : "--training-examples=3 -s --free-numbers",
            "hashing" : 2**15,
            #"ramdisk": "/dev/shm/yan",
            "learner" : learn.LightGBM(**learning)
         }
         
         models.check(settings)  
         log.start("Building LGB model:", settings)
         model = models.name(**settings) 
         rkeys = [(settings["bid"],pid,problem,settings["limit"]) for pid in settings["pids"] for problem in expres.benchmarks.problems(settings["bid"])]
         if not models.make(model, rkeys, settings):
            raise Exception("Enigma: FAILED: Building model %s" % model)
         efun = settings["learner"].efun()
         new.append(protos.solo(settings["pids"][0], model, mult=0, noinit=True, efun=efun))
         new.append(protos.coop(settings["pids"][0], model, mult=0, noinit=True, efun=efun))

log.msg("New strategies are available:\n%s\n" % "\n".join(new))

