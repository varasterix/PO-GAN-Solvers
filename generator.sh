#!/bin/bash

HIGHEST_WEIGHT="100"
TSP_FILE_PATH="data/tsp_files/"
SYMMETRIC=0

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

python ./generator_unit.py "$NB_CITIES" "$INSTANCE_ID" "$TSP_FILE_PATH" "$HIGHEST_WEIGHT" "SYMMETRIC"
