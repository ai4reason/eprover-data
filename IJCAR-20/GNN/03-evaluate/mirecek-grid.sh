#!/bin/sh

for QUERY in 32 64 128; do
   #for MODEL in g20k_20 g20k_50 g20k_99; do
   for MODEL in `ls models`; do
      for CONTEXT in 256 512 1024; do
         for BINARY in 0 1; do
            ./mirecek-instantiate.sh $QUERY $MODEL $CONTEXT $BINARY
         done
      done
   done
done

