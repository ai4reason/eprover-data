from datetime import datetime
from sys import argv, exc_info
import sys
import atexit

PREFIX = "~~~atpy~log~~~"

def terminating(cache):
   msg("Terminating.")
   if "last_traceback" in dir(sys):
      import traceback
      traceback.print_last(file=file(cache[1],"a"))


def start(intro, config=None, script=""):
   config = "\n   ".join(["%12s = %s"%(x,config[x]) for x in sorted(config)]) if config else ""
   intro = "%s\n\n   %s\n" % (intro, config)
   msg(intro, script=script, reset=True)

def msg(msg, cache=[], script="", timestamp=True, reset=False):
   now = datetime.now()
   if not cache:
      script = argv[0] if argv and not script else script
      cache.append(now)
      cache.append(("%s%s~~~%s" % (PREFIX, script.lstrip("./").replace("/","+"), now)).replace(" ","~"))
      atexit.register(terminating, cache)
   elif reset:
      cache[0] = now
   
   msg = ("[%s] %s" % (now-cache[0], msg)) if timestamp else msg
   print msg
   if PREFIX:
      f = file(cache[1],"a")
      f.write(msg+"\n")
      f.flush()
      f.close()

def text(msg0=""):
   msg(msg0, timestamp=False)

