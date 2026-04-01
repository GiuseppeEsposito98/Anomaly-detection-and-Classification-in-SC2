import os
import torch
import numpy as np
from tqdm import tqdm
import shutil

def get_argparser():
    parser = argparse.ArgumentParser(description='Supervised compression for image classification tasks')
    parser.add_argument('--dataset_dir', required=True, help='Directory containing the original simulation data')
    return parser

parser = get_argparser()
args = parser.parse_args()

BASE_DIR = f'{args.dataset_dir}/processed'

# CNF folders to process
CNF_FOLDERS = ['cnf1_JOBID0_W', 'cnf2_JOBID0_W', 'cnf3_JOBID0_W', 'cnf6_JOBID0_W', 'cnf9_JOBID0_W', 'cnf12_JOBID0_W']

# Data types to process
FM_ORIGIN = ['golden', 'faulty', 'AdversarialAttack']

# Output folder suffix for normalized data
OUTPUT_SUFFIX = 'processed'

# Clean existing normalized folders before processing
CLEAN_BEFORE_RUN = True  # Set to False to keep existing files


def clean_normalized_directories():
    """Remove all existing normalized directories"""
    print("=" * 80)
    print("Cleaning existing normalized directories...")
    print("=" * 80)
    
    cleaned_count = 0
    
    for cnf in CNF_FOLDERS:
        for split in ['train', 'val', 'test']:
            for data_type in FM_ORIGIN:
                # Define normalized directory to clean
                norm_dir = os.path.join(BASE_DIR, cnf, split, f"{data_type}{OUTPUT_SUFFIX}")
                
                if os.path.exists(norm_dir):
                    try:
                        shutil.rmtree(norm_dir)
                        print(f"  🗑️  Removed: {norm_dir}")
                        cleaned_count += 1
                    except Exception as e:
                        print(f"  ❌ Error removing {norm_dir}: {e}")
    
    if cleaned_count == 0:
        print("  No existing normalized directories found to clean.")
    else:
        print(f"\n✓ Cleaned {cleaned_count} directories")
    
    print("=" * 80 + "\n")


def compute_mean_std(folder):
    """Compute mean and std for a given folder"""
    all_tensors = []
    print(f"  Computing mean and std for: {folder}")
    
    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith(".pt"):
                tensor = torch.load(os.path.join(root, file))
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
                else:
                    tensor = tensor.to(dtype=torch.float32, device='cpu').detach()
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                all_tensors.append(tensor.view(tensor.shape[0], -1))  # Flatten each feature map along spatial dimensions

    if len(all_tensors) == 0:
        raise ValueError(f"No tensors found in {folder}")

    all_tensors = torch.cat(all_tensors, dim=1)  # Concatenate all tensors along the feature dimension
    mean = all_tensors.mean(dim=1, keepdim=True)
    std = all_tensors.std(dim=1, keepdim=True)
    std[std == 0] = 1.0  # Avoid division by zero
    
    print(f"  Mean shape: {mean.shape}, Std shape: {std.shape}")
    return mean, std


def normalize_folder(folder, save_folder, mean, std):
    """Normalize all tensors in a folder and save to new location"""
    os.makedirs(save_folder, exist_ok=True)
    
    file_count = 0
    for root, _, files in os.walk(folder):
        relative_path = os.path.relpath(root, folder)
        save_path = os.path.join(save_folder, relative_path)
        os.makedirs(save_path, exist_ok=True)

        for file in tqdm(files, desc=f"  Normalizing", leave=False):
            if file.endswith(".pt"):
                tensor = torch.load(os.path.join(root, file))
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
                else:
                    tensor = tensor.to(dtype=torch.float32, device='cpu').detach()
                tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Reshape mean/std from (C, 1) to (C, 1, 1) for broadcasting with (C, H, W)
                mean_reshaped = mean.view(-1, 1, 1)
                std_reshaped = std.view(-1, 1, 1)
                normalized_tensor = (tensor - mean_reshaped) / std_reshaped
                
                torch.save(normalized_tensor, os.path.join(save_path, file))
                file_count += 1
    
    return file_count


def process_cnf_folder(cnf_name):
    """Process a single CNF folder"""
    print(f"\n{'=' * 80}")
    print(f"Processing {cnf_name}")
    print(f"{'=' * 80}")
    
    cnf_path = os.path.join(BASE_DIR, cnf_name)
    
    if not os.path.exists(cnf_path):
        print(f"  ⚠️  CNF folder not found: {cnf_path}")
        return False
    
    # Process each data type
    for data_type in FM_ORIGIN:
        print(f"\n📁 Processing {data_type}:")
        
        # Compute mean and std from training data
        train_folder = os.path.join(cnf_path, "train", data_type)
        
        if not os.path.exists(train_folder):
            print(f"  ⚠️  Training folder not found: {train_folder}")
            continue
        
        try:
            mean, std = compute_mean_std(train_folder)
        except Exception as e:
            print(f"  ❌ Error computing mean/std: {e}")
            continue
        
        # Normalize all splits (train, val, test) using training statistics
        for split in ['train', 'val', 'test']:
            source_folder = os.path.join(cnf_path, split, data_type)
            save_folder = os.path.join(cnf_path, split, f"{data_type}{OUTPUT_SUFFIX}")
            
            if os.path.exists(source_folder):
                try:
                    file_count = normalize_folder(source_folder, save_folder, mean, std)
                    print(f"  ✓ {split}: Normalized {file_count} files → {save_folder}")
                except Exception as e:
                    print(f"  ❌ Error normalizing {split}: {e}")
            else:
                print(f"  ⚠️  {split} folder not found: {source_folder}")
    
    return True


def main():
    """Main function to process all CNF folders"""
    
    # Clean existing normalized directories if enabled
    if CLEAN_BEFORE_RUN:
        clean_normalized_directories()
    
    print("=" * 80)
    print("Starting feature normalization...")
    print("=" * 80)
    
    total_processed = 0
    total_skipped = 0
    
    for cnf in CNF_FOLDERS:
        try:
            success = process_cnf_folder(cnf)
            if success:
                total_processed += 1
            else:
                total_skipped += 1
        except Exception as e:
            print(f"\n❌ Error processing {cnf}: {e}")
            total_skipped += 1
    
    print("\n" + "=" * 80)
    print("Feature normalization completed!")
    print(f"Successfully processed: {total_processed} CNF folders")
    print(f"Skipped: {total_skipped} CNF folders")
    print("=" * 80)


if __name__ == "__main__":
    main()