#!/bin/bash -l

RUN_LOG="logs/training4mil_28core_$(date +%Y%m%d_%H%M%S).log"

javac -cp "./lib/*:." @pokePA.srcs

java -cp ".:./lib/*" edu.bu.pas.pokemon.ParallelTrain \
  src.pas.pokemon.agents.PolicyAgent \
  edu.bu.pas.pokemon.agents.RandomAgent \
  edu.bu.pas.pokemon.agents.AggroAgent \
  -p 5000 \
  -j 28 \
  -t 100 \
  -v 10 \
  -m 256 \
  -u 30 \
  -b 30000 \
  -r RANDOM \
  -g 0.99 \
  -d adam \
  -n 1.0E-4 \
  -o params/model_params4mil_28core 2>&1 | \
  tee "$RUN_LOG"