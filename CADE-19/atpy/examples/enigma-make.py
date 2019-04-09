#!/usr/bin/python

import sys
import os
from atpy import enigma

def make(name, f_pre):
   os.system("mkdir -p %s" % enigma.models.path(name))
   os.system("cp %s %s" % (f_pre, enigma.models.path(name, "train.pre")))
   enigma.models.smartboost(name)
   #enigma.models.standard(name)

if len(sys.argv) != 3:
   print "usage: %s model-name trains.pre" % os.path.basename(sys.argv[0])
else:
   make(sys.argv[1], sys.argv[2])



