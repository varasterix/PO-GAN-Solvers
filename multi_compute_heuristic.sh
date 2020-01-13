#!/bin/bash

TIME_LIMIT="10"
TSP_FILES_PATH="data/tsp_files/"
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
    -l)
    TIME_LIMIT="$2"
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
  echo "$TSP_FILES_PATH""dataSet_""$NB_CITIES""_$instance_id.heuristic"
  ./compute_heuristic.sh -n "$NB_CITIES" -i "$instance_id" -p "$TSP_FILES_PATH" -l "$TIME_LIMIT"
done
