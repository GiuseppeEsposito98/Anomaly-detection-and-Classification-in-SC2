import os
import random
import shutil
import torch
from pathlib import Path
import argparse
from config.py import *

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--dataset_dir', required=True, help='Directory containing the original simulation data')
    return parser

parser = get_argparser()
args = parser.parse_args()

BASE_DIR = f'{args.dataset_dir}'
CNF_FOLDERS = ['cnf1_JOBID0_W', 'cnf2_JOBID0_W', 'cnf3_JOBID0_W', 'cnf6_JOBID0_W', 'cnf9_JOBID0_W', 'cnf12_JOBID0_W']
FM_ORIGIN = ['faulty', 'AdversarialAttack']
data_split = ['train', 'val', 'test']

# Function to check if a .pt file contains NaNs
def contains_nan(file_path):
    try:
        data = torch.load(file_path)
        if isinstance(data, torch.Tensor):
            return torch.isnan(data).any().item()
        elif isinstance(data, dict):
            return any(torch.isnan(v).any().item() for v in data.values() if isinstance(v, torch.Tensor))
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return True  # Exclude files that cannot be loaded

for cnf in CNF_FOLDERS:
    cnf_dir = os.path.join(BASE_DIR, 'sampled', cnf)
    for data in FM_ORIGIN:
        data_path = os.path.join(cnf_dir, data)
        for split in data_split:
            split_path = os.path.join(data_path, split, 'sampled')
            nan_path = os.path.join(data_path, split, 'nan')
            samples = split_map[split]
            Path(split_path).mkdir(parents=True, exist_ok=True)
            Path(nan_path).mkdir(parents=True, exist_ok=True)

            original_data_path = os.path.join(BASE_DIR, cnf, 'hook_encoder_output', data)
            all_files = [f for f in os.listdir(original_data_path) if f.endswith(".pt")]

            nan_files = [f for f in all_files if contains_nan(os.path.join(original_data_path, f))]
            valid_files = [f for f in all_files if not contains_nan(os.path.join(original_data_path, f))]

            for _file in nan_files:
                src_path = os.path.join(original_data_path, _file)
                dest_path = os.path.join(nan_path, _file)
                shutil.copy2(src_path, dest_path)
                print(f"Moved {_file} to {nan_dir}")
            
            target_samples = min(samples, len(valid_files))

            # Randomly select files
            sampled_files = random.sample(valid_files, target_samples)

            # Copy sampled files to the new directory
            for file in sampled_files:
                src_path = os.path.join(original_data_path, file)
                dest_path = os.path.join(split_path, file)
                shutil.copy2(src_path, dest_path)  # copy2 preserves metadata
            
            print(f"Copied {target_samples} files to {sampled_dir}")
            print(f"Moved {len(nan_files)} files containing NaNs to {nan_dir}")
