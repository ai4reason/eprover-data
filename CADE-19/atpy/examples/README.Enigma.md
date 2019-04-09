**ATPy** for ENIGMA
===================

O Brave man!  You've come so far, don't let it slip away...

Warning
-------

Performance Monitoring Tools for Linux *perf* (http://perf.wiki.kernel.org) must be installed from your distribution repository.  The binary can not be provided, as it needs to match specific kernel version.  You also need reasonable permissons to access performance events system (say `1` in `/proc/sys/kernel/perf_event_paranoid`).

Quick start
-----------

### Warning ###

See the above warning!

### Quick, quick ... ###

```console
$ git clone https://github.com/ai4reason/atpy.git
$ cd atpy/examples
$ tar xzf enigma.tar.gz
$ cd enigma
$ ./example.sh
```

The example trains an Enigma predictor using accuracy-balancing boosting and
directed looping.  It evaluates the original and all the intermediate
predictors on a training set of 50 MZR problems.  It prints numbers of
processed clauses and the summary of solved problems.

Always run the example directly as above from the `enigma` directory and do
not mess about with the directory structure (unless you read the Usage).


Requirements
------------

Performance analysis tools for Linux *perf* must be installed on your
system.  Additionally, you need AI4Reason fork of E with Enigma support, and
LIBLINEAR.  You can use binaries provided in `bin/` compiled for x86 (you
still need to install *perf* from your distribution repository), or you
can obtain it from github, like this:

```console
$ git clone https://github.com/ai4reason/eprover.git
$ cd eprover
$ git fetch
$ git checkout ENIGMA
```

Then compile E and (included) LIBLINEAR:

```console
$ ./configure
$ make 
$ cd CONTRIB/liblinear
$ make
```

You need the following binaries in your `PATH` for **ATPy** to work:

```
PROVER/eprover
SIMPLE_APPS/enigma-features
CONTRIB/liblinear/train
CONTRIB/liblinear/predict
```


Installation
------------

Directory `atpy` must be added to your `PYTHONPATH`.


Usage
-----

Note: E _strategies_ are sometimes called _protocols_, from historical
reasons.

Once you have **ATPy** in PYTHONPATH, you can import it in Python:

```python
from atpy import *
```

Use `loop.py` and `example.sh` from `enigma.tar.gz` for an inspiration (see Quick start).

**ATPy** and an Enigma-inside E Prover make use of the following environment
variables which should be set appropriately.

+ `ENIGMA_ROOT`: Directory with Enigma models.
+ `EXPRES_PROTOS`: Directory with E strategies.
+ `EXPRES_BENCHMARKS`: Directory with benchmark problems.
+ `EXPRES_RESULTS`: Directory with E output files.
+ `EXPRES_SOLVED`: Directory with the database of solved problems for each
  strategy.

The following terminology is used in the code base:

+ `pid` is a _protocol id_, a file name relative to `EXPRES_PROTOS` which
  contains E command line options.  Do not use `/` in `pid` name.
  Additionally, `pid` can start with `Enigma+` and then it is looked up in
  `ENIGMA_ROOT` directory.
+ `pids` is list of `pid`s (`[pid]`).
+ `bid` is a _benchmark id_, a directory name relative `EXPRES_BENCHMARKS`
  which contains TPTP benchmark problems.  `bid` can contain `/`.
+ `results` is a collection (set, list, map, ...) of result keys (`rkey`),
  where each `rkey` is tuple `(pid,bid,problem,limit)` and it represents the
  results of running `pid` on `problem` from benchmark `bid` with time limit
  `limit`.  The corresponding E output file is in
  `EXPRES_RESULTS/bid/pid/limit`.

The following are the key function of **ATPy** for Enigma:

+ `expres.benchmarks.eval(bid, pids, limit)`: Evaluate strategies `pids` on
  benchmark `bid` with `limit`, and return `results` as a map from `rkeys`
  to parsed results.
+ `enigma.models.smartboost(name, results)`: Create model from training data
  `results`, use accuracy-balancing boosting, and put the result in
  `ENIGMA_ROOT/name`.
+ `enigma.protos.standalone(pid, name)`: Enrich strategy `pid` with Enigma
  model `name`, override any previous clause selection mechanism in `pid` to
  use Enigma predictor `name`.  Return new `pid`.
+ `enigma.protos.combined(pid, name)`: As above, but combine `name` with
  the clause selection from `pid`.
+ `expres.dump.solved(bid, pids, results, ref)`: Dump the number of problems
  from `bid` solved by each `pid` from `pids` with an optional reference
  `pid`.  Results must be already evaluated.




