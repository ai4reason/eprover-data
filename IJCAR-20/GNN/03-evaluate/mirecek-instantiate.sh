#!/bin/sh

if [ $# -ne 4 ]; then
	echo "usage: $0 QUERY MODEL CONTEXT BINARY"
	exit 1
fi

QUERY=$1
MODEL=$2
CONTEXT=$3
BINARY=$4

NAME="mzr02-${MODEL}-query${QUERY}-ctx${CONTEXT}-w${BINARY}"

cat mzr02-mirecek-solo.template | sed "s/QUERY/$QUERY/g" | sed "s/MODEL/$MODEL/g" \
	| sed "s/BINARY/$BINARY/g" | sed "s/CONTEXT/$CONTEXT/g" > strats/${NAME}-solo

cat mzr02-mirecek-coop.template | sed "s/QUERY/$QUERY/g" | sed "s/MODEL/$MODEL/g" \
	| sed "s/BINARY/$BINARY/g" | sed "s/CONTEXT/$CONTEXT/g" > strats/${NAME}-coop

