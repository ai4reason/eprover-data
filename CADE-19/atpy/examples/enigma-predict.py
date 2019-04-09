#!/usr/bin/python

import sys
import os
from atpy import enigma

PREDS = {
   "1" : "+",
   "10": "-"
}

def predict(name, f_cnfs, f_conj):
   os.system("rm -fr predict.pre predict.in predict.out")
   emap = enigma.enigmap.load(enigma.models.path(name, "enigma.map"))
   if not emap:
      raise Exception("Unknown or empty model %s" % name)

   enigma.pretrains.translate(f_cnfs, f_conj, "predict.pre")
   enigma.trains.make(file("predict.pre"), emap, file("predict.in","w"), strict=False)
   enigma.liblinear.predict("predict.in", enigma.models.path(name, "model.lin"), "predict.out", "/dev/null")

   cls = file(f_cnfs)
   prs = file("predict.out")

   while True:
      cl = cls.readline().strip()
      pr = prs.readline().strip()

      if (not cl) and (not pr):
         break
      elif (not cl) and pr:
         raise Exception("Classificator mismatch (too many predictions)")
      elif (not pr) and cl:
         raise Exception("Classificator mismatch (too many clauses)")

      print "%s|%s" % (PREDS[pr], cl)

   os.system("rm -fr predict.pre predict.in predict.out")

if len(sys.argv) != 4:
   print "usage: %s model-name classify.cnfs conjecture.cnfs" % os.path.basename(sys.argv[0])
else:
   predict(sys.argv[1], sys.argv[2], sys.argv[3])

