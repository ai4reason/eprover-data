#!/bin/sh

time ~/bin/eprover --tstp-format --soft-cpu-limit=50 --cpu-limit=51 -s --print-statistics --training-examples=1 --free-numbers --resources-info `cat $1` $2 2>&1

