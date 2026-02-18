#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"


mkdir -p ${global_PWD}/${DIR}

input_args=(1 2 3 6 9 12)

array_size=${#input_args[@]}


for ((i=0; i<$array_size; i++)); do
    bash ${global_PWD}/SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/01_DatasetGen/bash/AA_collection.sh ${input_args[$((i))]} ${DIR}
done