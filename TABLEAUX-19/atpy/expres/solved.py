import os
from .. import eprover

SOLVED_DIR = os.getenv("EXPRES_SOLVED", "./00SOLVED")

def path(bid, pid, limit):
   f_pid = "%s%s/%s" % (limit, "s" if isinstance(limit,int) else "", pid)
   return os.path.join(SOLVED_DIR, bid, f_pid)

def load(bid, pid, limit):
   f_solved = path(bid, pid, limit)
   if os.path.isfile(f_solved):
      return set(file(path(bid, pid, limit)).read().strip().split("\n"))
   else:
      return set()

def save(bid, pid, limit, problems):
   f_solved = path(bid, pid, limit)
   os.system("mkdir -p %s" % os.path.dirname(f_solved))
   file(f_solved, "w").write(("\n".join(sorted(problems)))+"\n")

def update(results):
   solved = {}
   for rkey in results:
      (bid, pid, problem, limit) = rkey
      if eprover.result.solved(results[rkey]):
         skey = (bid, pid, limit)
         if skey not in solved:
            solved[skey] = load(bid, pid, limit)
         solved[skey].add(problem)
   for skey in solved:
      (bid, pid, limit) = skey
      save(bid, pid, limit, solved[skey])

