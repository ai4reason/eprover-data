from .. import eprover, log

def make(bid, pids, results, ref=None):
   ret = {}
   
   if ref:
      ref = [r for r in results if r[0]==bid and r[1]==ref]
      ref = [r for r in ref if eprover.result.solved(results[r])]
      ref = [r[2:] for r in ref]
      ref = frozenset(ref)

   for pid in pids:
      problems = [r for r in results if r[0]==bid and r[1]==pid]
      total = len(problems)
      errors = [r for r in problems if eprover.result.error(results[r])]
      if errors:
         log.text("There were errors:\n%s" % "\n".join(map(str,errors)))
      errors = len(errors)
      solves = [r for r in problems if eprover.result.solved(results[r])]
      if ref:
         solves = [r[2:] for r in solves]
         solves = frozenset(solves)
         plus = len(solves-ref)
         minus = len(ref-solves)
      solves = len(solves)
      
      ret[pid] = [total, errors, solves]
      if ref:
         ret[pid] += [plus, minus]

   return ret

