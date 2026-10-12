#!/bin/bash
# Stops a training run if its resident memory exceeds a limit.
# usage: ./watchdog.sh <run name> [limit GB, default 16]
RUN=$1; LIMIT_KB=$(( ${2:-16} * 1024 * 1024 ))
sleep 20
while true; do
  P=$(pgrep -f "DQNTrainer out=runs/$RUN " | head -1)
  [ -z "$P" ] && exit 0
  R=$(ps -o rss= -p "$P" 2>/dev/null)
  if [ -n "$R" ] && [ "$R" -gt "$LIMIT_KB" ]; then
    echo "watchdog: killed pid $P at rss=${R}KB" >> "runs/$RUN/train.log"
    kill "$P"; exit 1
  fi
  sleep 10
done
