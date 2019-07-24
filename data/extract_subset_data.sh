#!/usr/bin/env bash

export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
export PROJECT_DIR="$(dirname "$PARENT_DIR")"
export DATA_DIR=$PROJECT_DIR/data

python extract_subset_data.py \
-d $DATA_DIR
