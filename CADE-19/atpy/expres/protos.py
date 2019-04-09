import os

PROTOS_DIR = os.getenv("EXPRES_PROTOS", "strats")

def path(pid):
   if pid.startswith("Enigma+"):
      pid = pid.replace("+", "/")[7:]
      root = os.getenv("ENIGMA_ROOT", "./Enigma")
   else:
      root = PROTOS_DIR
   return os.path.join(root, pid)

def load(pid):
   return file(path(pid)).read().strip()

def save(pid, proto):
   f_pid = path(pid)
   os.system("mkdir -p %s" % os.path.dirname(f_pid))
   file(f_pid,"w").write(proto.strip())

