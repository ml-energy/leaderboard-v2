#!/usr/bin/env bash

docker run \
  --name controller \
  --net leaderboard \
  -v $HOME/workspace/leaderboard:/workspace/leaderboard \
  -v $HOME/workspace/text-generation-inference/deployment:/workspace/text-generation-inference/deployment:ro \
  -v /data/leaderboard/colosseum-controller-logs:/logs \
  -p 7778:8000 \
  -e LOG_DIR=/logs \
  mlenergy/colosseum-controller:latest
