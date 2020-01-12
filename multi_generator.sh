#!/bin/bash

HIGHEST_WEIGHT="100"
TSP_FILES_PATH="data/tsp_files/"
SYMMETRIC=0

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n)
    NB_CITIES="$2"
    shift
    ;;
    -x)
    NB_INSTANCES="$2"
    shift
    ;;
    -p)
    TSP_FILES_PATH="$2"
    shift
    ;;
    -h)
    HIGHEST_WEIGHT="$2"
    shift
    ;;
    -s)
    SYMMETRIC="$2"
    shift
    ;;
    *)
        echo "Argument inconnu: ${1}"
        exit
    ;;
esac
shift
done

# shellcheck disable=SC2004
for ((instance_id = 0; instance_id < $NB_INSTANCES; instance_id++))
do
  echo "$TSP_FILES_PATH""dataSet_""$NB_CITIES""_$instance_id.tsp"
  ./generator.sh -n "$NB_CITIES" -i "$instance_id" -p "$TSP_FILES_PATH" -h "$HIGHEST_WEIGHT" -s "$SYMMETRIC"
done
