#!/bin/bash
# SBATCH options
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --partition=<avail_partition>
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=NeuronsSplit

# 1 Activate the virtual environment
source ~/miniconda3/bin/activate
conda deactivate

cd ~/AlternativeModels-SC2
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
        --device cuda > ${global_PWD}/${DIR}/cnf${target_config}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_stde.log

echo
echo "All done. Checking results:"