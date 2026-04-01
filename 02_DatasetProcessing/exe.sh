#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd ~/AlternativeModels-SC2
conda activate sc2-benchmark-fsim


PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0

DIR="$1"

Sim_dir=${global_PWD}/${DIR}/


python ${global_PWD}/SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/02_DatasetProcessing/sample_normalize_golden.py \
        --dataset_dir ${Sim_dir}

python ${global_PWD}/SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/02_DatasetProcessing/sample.py \
        --dataset_dir ${Sim_dir}
    
python ${global_PWD}/SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/02_DatasetProcessing/normalization.py \
        --dataset_dir ${Sim_dir}
