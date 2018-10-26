#!/bin/sh
data=$1

libbi filter \
  --model-file RandomWalk.bi \
  --obs-file $data \
  --filter bootstrap \
  --nparticles 20000 \
  --start-time 0 \
  --end-time $2 \
  --output-file ../filtered.nc
