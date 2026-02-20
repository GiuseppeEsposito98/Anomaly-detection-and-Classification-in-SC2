import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_loader_sample import get_dataloaders
import optuna
from model import FeatureMapClassifier
from utils import *

def main():
    parser = argparse.ArgumentParser(description='Train Feature Map Classifier')
    parser.add_argument('--data_root', type=str, default='../dataset/',
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--train_portion', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--statistic_dir', type=str, default='runs', help='Directory to save the runs statistic')
    
    args = parser.parse_args()
    
    # Print run configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Data root:      {args.data_root}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*50 + "\n")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Get data loaders
    print("\nLoading datasets...")
    subset_ratios = {"train": float(args.train_portion), "val": 1 , "test": 1}
    print(f'train portion: {args.train_portion}')

    dataloaders = get_dataloaders(args.data_root, batch_size=args.batch_size, subset_ratios=subset_ratios)

    # Get a sample batch to determine input shape
    for inputs, labels, _ in dataloaders['train']:
        sample_inputs = inputs.squeeze(1)  # Remove the extra dimension
        sample_labels = labels
        break
    
    input_shape = sample_inputs.shape[1:]  # (C, H, W)
    print(f'Input feature map shape: {input_shape}')
    print(f'Sample batch shape: {sample_inputs.shape}')
    print(f'Sample labels: {sample_labels[:10]}')  # Show first 10 labels

    # Create model
    print("\nCreating model...")


    # Create a study and optimize the objective function

    sampler = set_random_seed(42)

    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial):
        # Define hyperparameters to tune
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        dropout = trial.suggest_uniform('dropout', 0.1, 0.9)
        hidden_dimension_1 = trial.suggest_int('hidden_dimentsion_1', 16, 256, step=16)
        hidden_dimension_2 = trial.suggest_int('hidden_dimentsion_2', 8, 128, step=8)
        optimizer_type = trial.suggest_categorical('optimizer_type', ['Adam', 'SGD', 'RMSprop'])
        scheduler_type = trial.suggest_categorical('scheduler_type', ['StepLR', 'CosineAnnealingLR'])

        # Create model with the suggested dropout
        model = FeatureMapClassifier(input_dim=input_shape[0]*input_shape[1]*input_shape[2], 
        hidden_dim_1=hidden_dimension_1, 
        hidden_dim_2=hidden_dimension_2, 
        dropout=dropout)

        model = model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()

        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)

        if optimizer_type == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_type == 'SGD':
            momentum = trial.suggest_uniform('momentum', 0.0, 0.9)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'RMSprop':
            alpha = trial.suggest_uniform('alpha', 0.9, 0.99)
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, alpha=alpha, weight_decay=weight_decay)

        # Define hyperparameters to tune for the scheduler

        # Learning rate scheduler
        if scheduler_type == 'StepLR':
            step_size = trial.suggest_int('step_size', 5, 50, step=5)
            gamma = trial.suggest_uniform('gamma', 0.1, 0.9)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'ReduceLROnPlateau':
            factor = trial.suggest_uniform('factor', 0.1, 0.9)
            patience = trial.suggest_int('patience', 1, 10)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience)
        elif scheduler_type == 'CosineAnnealingLR':
            T_max = trial.suggest_int('T_max', 10, 100)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        
        unique_id = f'learning_rate_{learning_rate}_dropout_{dropout}_hidden1_{hidden_dimension_1}_hidden2_{hidden_dimension_2}_opt_{optimizer_type}_sched_{scheduler_type}_trial_{trial.number}'
        summWriter = SummaryWriter(log_dir=f"{args.statistic_dir}_{args.train_portion}/{args.data_root.split('/')[2]}/run{trial.number}")
        # print(unique_id)
        # Train model
        history = train_model(
            model, dataloaders, criterion, optimizer, device,
            num_epochs=args.num_epochs, checkpoint_dir=f"{args.checkpoint_dir}/{args.data_root.split('/')[2]}", log_dir=f"{args.statistic_dir}/{args.data_root.split('/')[2]}", unique_id=unique_id,
            summWriter=summWriter, scheduler=scheduler
        )

        # Return the best validation accuracy
        return history['best_val_acc']
    
    study.optimize(objective, n_trials=10)


    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)
    model = FeatureMapClassifier(input_dim=input_shape[0]*input_shape[1]*input_shape[2], 
        hidden_dim_1=study.best_params['hidden_dimentsion_1'], 
        hidden_dim_2=study.best_params['hidden_dimentsion_2'], 
        dropout=study.best_params['dropout'])
    print(study.best_params.keys())
    model.load_state_dict(torch.load(f"./checkpoints/{args.data_root.split('/')[2]}/best_model_learning_rate_{study.best_params['learning_rate']}_dropout_{study.best_params['dropout']}_hidden1_{study.best_params['hidden_dimentsion_1']}_hidden2_{study.best_params['hidden_dimentsion_2']}_opt_{study.best_params['optimizer_type']}_sched_{study.best_params['scheduler_type']}_trial_{study.best_trial.number}.pth"))
    model = model.to(device)
    evaluate_model(
        model, dataloaders['test'], device = device, unique_id='final_model', criterion = nn.CrossEntropyLoss()
    )

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print(model)
    

if __name__ == '__main__':
    main()