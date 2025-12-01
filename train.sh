#!/bin/bash -l
#$ -N pokemon_train
#$ -P cs440
#$ -l h_rt=168:00:00
#$ -pe omp 8
#$ -l mem_per_core=8G
#$ -j y
#$ -m ea

SELFPLAY_DIR=${SELFPLAY_DIR:-"params"}
SELFPLAY_METRICS=${SELFPLAY_METRICS:-"logs/selfplay_history.log"}
SELFPLAY_K=${SELFPLAY_K:-30}
RUN_LOG="logs/training_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$(dirname "$SELFPLAY_METRICS")"
mkdir -p "$(dirname "$RUN_LOG")"

javac -cp "./lib/*:." @pokePA.srcs
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

java -Xmx12g -XX:+UseG1GC \
  -Dselfplay.dir="$SELFPLAY_DIR" \
  -Dselfplay.metrics="$SELFPLAY_METRICS" \
  -Dselfplay.k="$SELFPLAY_K" \
  -cp ".:./lib/*" edu.bu.pas.pokemon.Train \
  src.pas.pokemon.agents.PolicyAgent \
  src.pas.pokemon.agents.SelfPlayPolicyAgent \
  edu.bu.pas.pokemon.agents.RandomAgent \
  edu.bu.pas.pokemon.agents.AggroAgent \
  edu.bu.pas.pokemon.agents.Drac5290Agent \
  -p 2000 \
  -t 15 \
  -v 5 \
  -m 32 \
  -u 50 \
  -b 20000 \
  -r OLDEST \
  -g 0.99 \
  -o params/model_params 2>&1 | \
  tee "$RUN_LOG" | tee -a "$SELFPLAY_METRICS"