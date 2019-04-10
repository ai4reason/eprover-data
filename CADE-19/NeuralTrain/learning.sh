#!/bin/bash

mkdir datasets
mkdir logs

tar xvf train_cnf.tgz

if hash pypy3 2>/dev/null; then
    pypy3 parser-pypy.py --dirs train_cnf/ --problem train_cnf/cnf/ datasets/paper-bushy-mzr02-10s
else
    python3 parser-pypy.py --dirs train_cnf/ --problem train_cnf/cnf/ datasets/paper-bushy-mzr02-10s
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 ennigma.py \
	--batch 128 \
	--batch_problem \
	--clause_depth 1 \
	--comb_type LinearReLU6 \
	--conjecture \
	--conjecture_depth 1 \
	--conjecture_dim 16 \
	--dev cuda:0 \
	--dim 64 \
	--epoch 50 \
	--final_type 3LinearReLU \
	--fnc_depth 1 \
	--no-full_model \
	--lr 0.001 \
	--min_occurr 10 \
	--no-negative_as_positive \
	--positive 1.0 \
	--pred_depth 1 \
	--random_order \
	--split_problem \
	--log logs/ \
	--log_filename output.log \
	datasets/paper-bushy-mzr02-10s

python3 export-model.py logs/output.log.49 model
