import os
import json

def save(f_js, var, header, classes, rows, leg=None):
   os.system("mkdir -p %s" % os.path.dirname(f_js))
   js = {}
   js["HEADER"] = header
   js["CLASSES"] = classes
   js["DATA"] = rows
   if leg:
      js["LEGEND"] = leg
   file(f_js,"w").write("var %s = %s;" % (var,json.dumps(js)))

def load(f_js):
   js = file(f_js).read().strip().rstrip(";")
   js = js[js.index("=")+1:]
   return json.loads(js)

def update(f_js, var, rows, key=None):
   js = load(f_js)
   if not key:
      key = lambda x: x

   new = {key(r):r for r in rows}
   toadd = list(new.keys())

   if len(rows) != len(new):
      raise Exception("Keys are not unique!")

   olds = js["DATA"]
   for i in xrange(len(olds)):
      row = olds[i]
      k = key(row)
      if k in new:
         olds[i] = new[k]
         toadd.remove(k)
   for k in toadd:
      olds.append(new[k])

   leg = js["LEGEND"] if "LEGEND" in js else None
   save(f_js, var, js["HEADER"], js["CLASSES"], olds, leg)

