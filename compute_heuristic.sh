#!/bin/bash

TIME_LIMIT="10"
TSP_FILE_PATH="data/tsp_files/"
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n)
    NB_CITIES="$2"
    shift
    ;;
    -i)
    INSTANCE_ID="$2"
    shift
    ;;
    -p)
    TSP_FILE_PATH="$2"
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

python ./compute_heuristic_unit.py "$NB_CITIES" "$INSTANCE_ID" "$TSP_FILE_PATH" "$TIME_LIMIT"
