#!/bin/bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 <file.smv>"
  exit 1
fi

SMV_FILE=$1

../../Tools/nuXmv/nuXmv -source /dev/stdin "$SMV_FILE" <<EOF
read_model -i $SMV_FILE
flatten_hierarchy
encode_variables
build_boolean_model
check_ltlspec_ic3
quit
EOF
