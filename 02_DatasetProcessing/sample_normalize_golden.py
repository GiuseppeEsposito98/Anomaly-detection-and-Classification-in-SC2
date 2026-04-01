import os
import torch
import shutil
from tqdm import tqdm
import argparse
from config import *

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--dataset_dir', required=True, help='Directory containing the original simulation data')
    return parser

parser = get_argparser()
args = parser.parse_args()

BASE_DIR = f'{args.dataset_dir}'
CNF_FOLDERS = ['cnf1_JOBID0_W', 'cnf2_JOBID0_W', 'cnf3_JOBID0_W', 'cnf6_JOBID0_W', 'cnf9_JOBID0_W', 'cnf12_JOBID0_W']
FM_ORIGIN = ['golden', 'faulty', 'AdversarialAttack']
data_split = ['train', 'val', 'test']

def compute_mean_std(folder):
    all_tensors = []
    print(f"Computing mean and std for folder: {folder}")
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".pt"):
                tensor = torch.load(os.path.join(root, file))
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32, device='cuda')
                else:
                    tensor = tensor.to(dtype=torch.float32, device='cuda').detach()
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                all_tensors.append(tensor.view(tensor.shape[0], -1))
    if len(all_tensors) == 0:
        raise ValueError(f"No tensors found in {folder}")
    all_tensors = torch.cat(all_tensors, dim=1)
    mean = all_tensors.mean(dim=1, keepdim=True)
    std = all_tensors.std(dim=1, keepdim=True)
    std[std == 0] = 1.0  # Avoid division by zero
    print(f"Mean computed: {mean.squeeze().tolist()}")
    print(f"Std computed: {std.squeeze().tolist()}")
    return mean, std

def normalize_tensor(tensor, mean, std):
    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    return (tensor - mean) / std

def split_and_normalize_golden(folder, output_root_folder, split_sizes):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])
    total_samples = len(files)
    print(total_samples)
    print(len(files))
    # assert sum(split_sizes) == total_samples, "Split sizes do not match the number of samples"
    
    splits = ["train", "val", "test"]
    start_idx = 0
    
    # First, separate files into different splits
    split_files = {}
    for split, size in zip(splits, split_sizes):
        split_files[split] = files[start_idx:start_idx + size]
        start_idx += size
    
    # Create temporary folder for training files
    train_temp_folder = os.path.join(output_root_folder, "train_temp")
    os.makedirs(train_temp_folder, exist_ok=True)
    
    # Copy training files to temporary folder
    for file in tqdm(split_files["train"], desc="Copying train files to temp"):
        src_path = os.path.join(folder, file)
        dst_path = os.path.join(train_temp_folder, file)
        tensor = torch.load(src_path)
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor, dtype=torch.float32, device='cuda')
        else:
            tensor = tensor.to(dtype=torch.float32, device='cpu').detach()
        torch.save(tensor, dst_path)
    
    # Compute mean and std only from the training set
    mean, std = compute_mean_std(train_temp_folder)
    
    # Process each split
    for split in splits:
        print(f"Processing {split} with {len(split_files[split])} samples")
        split_folder = os.path.join(output_root_folder, split, "golden")
        os.makedirs(split_folder, exist_ok=True)
        
        for file in tqdm(split_files[split], desc=f"Normalizing {split}"):
            tensor = torch.load(os.path.join(folder, file))
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor, dtype=torch.float32, device='cuda')
            else:
                tensor = tensor.to(dtype=torch.float32, device='cpu').detach()
            tensor = normalize_tensor(tensor, mean, std)
            torch.save(tensor, os.path.join(split_folder, file))
    
    # Clean up temporary folder
    shutil.rmtree(train_temp_folder)

if __name__ == "__main__":
    # input_folder = r"C:/Users/scara/repositories/split_computing_feature_analysis_and_detection/dataset/raw/FSIM_N_HPC_img_Giuseppe_comp_gold/cnf9_JOBID0_W/hook_encoder_output/ComprehensiveGolden"
    # output_root_folder = r"C:/Users/scara/repositories/split_computing_feature_analysis_and_detection/dataset/test_train_9_norm"
    for cnf in CNF_FOLDERS:
        input_folder = os.path.join(BASE_DIR, cnf, "hook_encoder_output", "golden")
        output_root_folder = os.path.join(BASE_DIR, 'processed', cnf, 'golden')
        split_and_normalize_golden(input_folder, output_root_folder, split_sizes)