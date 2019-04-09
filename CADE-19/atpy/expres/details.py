from .. import eprover

def make(bid, pids, results, selector=None, key=None, none="-"):
   details = {}
   if not selector:
      selector = lambda result: result[key] if key in result else none
   problems = [(problem,limit) for (bid,pid,problem,limit) in results]
   #pid0 = pids[0]
   for (problem,limit) in problems:
      for pid in pids:
         rkey = (bid,pid,problem,limit)
         if (problem,limit) not in details:
            details[(problem,limit)] = {}
         details[(problem,limit)][pid] = selector(results[rkey]) if rkey in results and eprover.result.solved(results[rkey])else none
         #if pid != pid0 and pid0 in details[(problem,limit)]:
         #   if type(details[(problem,limit)][pid]) is int and \
         #      type(details[(problem,limit)][pid0]) is int:
         #         details[(problem,limit)][pid] = "%.1f" % (\
         #            float(details[(problem,limit)][pid0]) / \
         #            float(details[(problem,limit)][pid]))
   return details

def processed(bid, pids, results):
   return make(bid, pids, results, key="PROCESSED")

