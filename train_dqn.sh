#!/bin/bash
# Double DQN (3-step returns) + self-play trainer (src/pas/pokemon/train/DQNTrainer.java).
# usage: ./train_dqn.sh <run name> [extra key=value overrides...]
# Pair it with ./watchdog.sh <run name> 18 to stop the run if memory runs away.
set -e
cd "$(dirname "$0")"
JAVA_HOME=${JAVA_HOME:-$HOME/tools/jdk8/jdk8u504-b01}
RUN=${1:-dqn}; shift || true
mkdir -p build "runs/$RUN"
"$JAVA_HOME/bin/javac" -nowarn -d build -cp "./lib/*" @pokePA.srcs @tools.srcs
# patches/pokePA-patches.jar must come before lib/* so the fixed framework classes win
nice -n 5 "$JAVA_HOME/bin/java" -Xmx10g -XX:+UseParallelGC \
  -Djava.util.concurrent.ForkJoinPool.common.parallelism=16 \
  -cp "build:patches/pokePA-patches.jar:lib/*" src.pas.pokemon.train.DQNTrainer \
  out="runs/$RUN" threads=16 hours=8 gamesPerCycle=256 \
  buffer=200000 warmup=20000 batch=256 replayRatio=4 \
  gamma=0.97 nStep=3 lr=2e-4 lrEnd=5e-5 huber=1.0 targetSync=1000 \
  epsStart=1.0 epsEnd=0.05 epsDecayGames=100000 \
  pDrac=0.03 dracTrainMaxTeam=2 pRandom=0.15 pAggro=0.30 pTeam2=0.5 \
  oppEps=0.05 snapshotEvery=25 poolSize=16 \
  minTeam=1 maxTeam=6 evalEvery=10 evalPairs=500 \
  dracEvalEvery=100 dracEvalPairs=64 dracMaxTeam=3 checkpointEvery=200 \
  "$@" 2>&1 | tee -a "runs/$RUN/train.log"
