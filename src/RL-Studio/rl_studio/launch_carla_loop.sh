#!/bin/bash

# Path to your CARLA startup script
carla_startup_script="/opt/0913/carla-simulator/CarlaUE4.sh"

if [[ -z "$2" ]]; then
    params="--world-port=$1 -opengl -RenderOffScreen -nosound"
else
    params="--world-port=$1 -nosound"
fi

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting CARLA server... at port $1"

  "$carla_startup_script" $params
  exit_code=$?

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] CARLA server exited with code: $exit_code"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Restarting CARLA server..."
done

