# AI-Based Detection and Classification of Adversarial and Fault-Induced Threat in Split Computing

This repository implements the methodology developed to distinguish *adversarial attacks* from *hardware‑induced fault corruptions* in **Split Computing (SC)** systems.  
In SC architectures, DNNs are partitioned between an edge device and the cloud, exposing them to both adversarial perturbations and hardware computation faults. These two error sources can produce similar misprediction patterns, making them difficult to differentiate.

We propose a unified framework that generates adversarial examples, simulates hardware faults on split models, and trains classifiers to determine the underlying cause of a misprediction—without modifying the DNN or SC execution pipeline.

---

## Requirements

First install **Miniconda (Python 3.8)**:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b
```

## Installation
1. Clone the repository
This repository is a fork of the original sc2-benchmark, pinned to a commit compatible with our framework.
All credits for the original benchmark belong to: https://github.com/yoshitomo-matsubara/sc2-benchmark

```bash
git clone https://github.com/GiuseppeEsposito98/AlternativeModels-SC2
cd AlternativeModels-SC2
```

2. Download and decompress the pre-trained Split DNN models
Follow the instructions inside the benchmark directory if needed.

3. Install SC_Fault_injections

```bash
git clone https://github.com/divadnauj-GB/SC_Fault_injections
cd SC_Fault_injections
find . -name "*.sh" | xargs chmod +x
```

4. Install Pytorchfi_SC
```bash
git clone https://github.com/divadnauj-GB/pytorchfi_SC
```

5. Create the sc2-benchmark environment

```bash

cp environment.yaml ../environment.yaml
cd ..
conda deactivate

conda env create -f environment.yaml
conda deactivate
source ~/miniconda3/bin/activate sc2-benchmark-fsim
```

6. Install dependencies

```bash
python -m pip install -e .

python -m pip install -e ./SC_Fault_injections/pytorchfi_SC/
```

7. Install anomaly‑detection and classification repo and dependencies

```bash
git clone https://github.com/GiuseppeEsposito98/Anomaly-detection-and-Classification-in-SC2.git
cd ./SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2
pip install -r requirements
cd ../..
```

## Usage
The experiments might require time, so my suggestion is to use an High Performance Computing (HPC) system to speedup the process.
If you want to run on an HPC, 
1. properly setup the sbatch file in SC_Fault_injections/01_DatasetGen/SLURM_scripts folder 
2. run the following command.
```bash
bash SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/01_DatasetGen/SLURM_script/Collect_data_security.sh SIM_data
bash SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/01_DatasetGen/SLURM_script/Collect_data_security.sh SIM_data
```
Otherwise you can run in local with this command
```bash
bash SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/01_DatasetGen/bash/Collect_data_security.sh SIM_data
bash SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/01_DatasetGen/bash/Collect_data_reliability.sh SIM_data
```
You will find the collected feature maps in the folder SIM_data

Then you can run the code for Feature maps postprocessing:
```bash
<postprocessed_data_folder>
```

Once you have the data saved in <postprocessed_data_folder> you can run the classifier training:
```bash
cd ./SC_Fault_injections/Anomaly-detection-and-Classification-in-SC2/03_AnomalyDetectionAndClassification
bash run_configuration.sh <postprocessed_data_folder>
```

## Citation
If you use this repository, framework, or methodology in your research, please cite:
https://ieeexplore.ieee.org/abstract/document/11116959

```bash

@inproceedings{esposito2025ai,
  title={AI-Based Classification of Adversarial Attacks vs. Hardware Fault Corruptions in the Split Computing Context},
  author={Esposito, G and Magliano, E and Scarano, N and Eltaras, T Ahmed and Balaguera, JD Guerrero and Mannella, L and Condia, JE Rodriguez and Ruospo, A and Di Carlo, S and Levorato, M and others},
  booktitle={2025 IEEE 31st International Symposium on On-Line Testing and Robust System Design (IOLTS)},
  pages={1--7},
  year={2025},
  organization={IEEE}
}
```

## References & Acknowledgments
This work builds upon and integrates the following open‑source projects:

### sc2-benchmark
Original Split Computing benchmark
https://github.com/yoshitomo-matsubara/sc2-benchmark
Our fork exists only to pin a compatible commit. Full credit belongs to the original authors.

### SC_Fault_injections
Fault injection framework for Split Computing
https://github.com/divadnauj-GB/SC_Fault_injections