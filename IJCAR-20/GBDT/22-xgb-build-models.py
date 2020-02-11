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

#learning = {
#   #'learning_rate': 0.2, 
#   'learning_rate': 0.3, 
#   #'num_leaves': 600, 
#   'num_leaves': 900, 
#   #'num_round': 150, 
#   'num_round': 150, 
#   'reg_lambda': 0, 
#   'random_state': 42, 
#   'objective': 'binary', 
#   'min_child_samples': 40, 
#   #'max_depth': 16,
#   'max_depth': 20,
#}

new = []

for ETA in [0.2]: #[0.2, 0.3]:
   for DEPTH in [12]: #[12, 9, 16, 20]:

      learning = {
            'max_depth': DEPTH, # 9. 12, 16, 20?
            'num_round': 150,
            'eta': ETA,
            'tree_method':'gpu_hist',
            'n_gpus': 4
            #learning_rate=0.2,
            #boost_from_average=False,
            #feature_fraction=0.8,
            #bagging_fraction=0.8,
            #min_data_in_leaf=45,
      }
      
      settings = {
         "bid"     : "mizar40/all",
         "pids"    : ['mzr02','Enigma+mizar40-all-T10+mzr02-VHSLCAXPh+xgb-d12-e0.2+coop','Enigma+mizar40-all-T10+mzr02-VHSLCAXPh+xgb-d12-e0.2+solo'],
         "ref"     : "loop02",
         "limit"   : "T10",
         "cores"   : 60,
         "version" : "VHSLCAXPh",
         #"eargs"   : "--training-examples=3 -s --free-numbers",
         "hashing" : 2**15,
         #"ramdisk": "/dev/shm/yan",
         "gzip"    : False,
         "learner" : learn.XGBoost(**learning)
      }
      
      models.check(settings)  
      log.start("Building XGB models:", settings)
      model = models.name(**settings)
      rkeys = [(settings["bid"],pid,problem,settings["limit"]) for pid in settings["pids"] for problem in expres.benchmarks.problems(settings["bid"])]
      if not models.make(model, rkeys, settings):
         raise Exception("Enigma: FAILED: Building model %s" % model)
      efun = settings["learner"].efun()
      new.append(protos.solo(settings["pids"][0], model, mult=0, noinit=True, efun=efun))
      new.append(protos.coop(settings["pids"][0], model, mult=0, noinit=True, efun=efun))

log.msg("New strategies are available:\n%s\n" % "\n".join(new))

