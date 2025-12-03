#!/bin/bash -l
#$ -N pokemon_train
#$ -P cs440
#$ -l h_rt=100:00:00
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -m ea

RUN_LOG="logs/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$RUN_LOG")"

javac -cp "./lib/*:." @pokePA.srcs
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

java -Xmx20g -XX:+UseG1GC -XX:MaxGCPauseMillis=200 \
  -XX:ParallelGCThreads=8 \
  -cp ".:./lib/*" edu.bu.pas.pokemon.Train \
  src.pas.pokemon.agents.PolicyAgent \
  edu.bu.pas.pokemon.agents.RandomAgent \
  edu.bu.pas.pokemon.agents.AggroAgent \
  edu.bu.pas.pokemon.agents.Drac5290Agent \
  -p 2000 \
  -t 15 \
  -v 1 \
  -m 256 \
  -u 1 \
  -b 5000 \
  -r OLDEST \
  -g 0.99 \
  -d adam \
  -n 1.0E-3 \
  -o params/model_params 2>&1 | \
  tee "$RUN_LOG"