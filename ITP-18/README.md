This repository contains data for the paper ProofWatch: Watchlist Guidance for Large Theories in E (https://arxiv.org/abs/1802.04007) submitted to the Interactive Theorem Proving 2018 conference.

The experimental data is in ITP-results.
The E prover strategies run are in strategies.

The E prover code can be found on the github repository https://github.com/JUrban/eprover/tree/ITP18.

Specifically at this time (May 15, 2018), the PCL-watchlists branch (https://github.com/JUrban/eprover/tree/PCL-watchlists) can be used for the following strategies: baseline, const, pref, dyn, and evo.
For ska (matching skolem symbols modulo their name), the branch https://github.com/JUrban/eprover/tree/SKOLEM-MATCH-DECAY can be used.
For dyndec (dynamic watchlist matching with relevance inheritance, the branch https://github.com/JUrban/eprover/tree/ITP18-decay1 can be used. The ska branch is also usable.
(Strategy uwl used a private version of E and was not ported to this repo due to poor performance.)


## How To Run The Code

0. Compile E prover.

1. Generate proofs by using eprover-any.sh with the strategy and problem as input (correcting for your eprover binary location). This is best done using GNU Parallel (https://www.gnu.org/software/parallel) or the scripting language of your choice.

2. Extract "trainpos" clauses from the output files. These can be used as hints on watchlist files.
  
3. 
..* To test a static watchlist method, put hints from step 2 on to one file and add "--watchlist=$3" to the script where $3 is the watchlist file.
..* To test a dynamic watchlist method, our suggestion is to put a watchlist file of hints for each desired proof in one directory, and add "--watchlist-dir=$3" where $3 is the watchlist directory.
..* To use the matching modulo skolem symbol name feature add "--wl-normalize-skolem" as an option to the appropriate E prover binary.


