#!/bin/bash

source ~/anaconda3/bin/activate
conda deactivate

cd ~/Desktop/Ph.D_/projects/SC2_privacy_reliability/code/AlternativeModels-SC2
conda activate sc2-benchmark-fsim


PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}

job_id=0

target_config="$1"
DIR="$2"


Sim_dir=${global_PWD}/${DIR}/cnf${target_config}_JOBID${job_id}_W
mkdir -p ${Sim_dir}


cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${target_config}ch_from_resnet50.yaml ${Sim_dir}
sed -i "s+ckpt: !join \['./resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+g" ${Sim_dir}/resnet50-bq${target_config}ch_from_resnet50.yaml

cd ${Sim_dir}

python ${global_PWD}/SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/01_DatasetGen/image_classification_AA.py \
        --config ${Sim_dir}/resnet50-bq${target_config}ch_from_resnet50.yaml\
        --device cuda \
        -aa > ${global_PWD}/${DIR}/cnf${target_config}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_stde.log

echo
echo "All done. Checking results:"