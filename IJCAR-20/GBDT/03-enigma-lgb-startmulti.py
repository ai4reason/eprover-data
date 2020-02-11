#!/usr/bin/env python3

from pyprove import log, expres
from enigmatic import models, learn

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

learning = {
   #'learning_rate': 0.2, 
   'learning_rate': 0.3, 
   #'num_leaves': 600, 
   'num_leaves': 900, 
   #'num_round': 150, 
   'num_round': 150, 
   'reg_lambda': 0, 
   'random_state': 42, 
   'objective': 'binary', 
   'min_child_samples': 40, 
   #'max_depth': 16,
   'max_depth': 20,
}

settings = {
   "bid"     : "mizar40/all",
   "pids"    : open("greedy.5b").read().strip().split("\n"),
   "ref"     : "mzr05",
   "limit"   : "T5",
   "cores"   : 60,
   #"version" : "VHSLCAXPh",
   "version" : "VHSLCXPh",
   "eargs"   : "--training-examples=3 -s --free-numbers",
   "hashing" : 2**15,
   "ramdisk": "/dev/shm/yan",
   "learner" : learn.LightGBM(**learning)
}
  
log.start("Starting Enigma experiments:", settings)

ref = settings["ref"]
settings["ref"] = "greedy5b-"+ref
model = models.name(**settings)
settings["ref"] = ref

for n in range(10):
   models.loop(model, settings, nick="loop%02d"%n)
   expres.dump.solved(**settings)

expres.dump.processed(**settings)
expres.dump.solved(**settings)

