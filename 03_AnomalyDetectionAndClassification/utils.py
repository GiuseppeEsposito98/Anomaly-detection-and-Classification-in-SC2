
import os
import time
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import csv
from optuna.samplers import TPESampler

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, checkpoint_dir="checkpoints", max_grad_norm=1.0, log_dir=None, unique_id=None, summWriter=None, scheduler=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize training history
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': [],
        'best_val_acc': 0.0, 'best_epoch': 0
    }
    
    # Get dataset sizes
    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in ['train', 'val', 'test']}
    print(f"Dataset sizes: Train: {dataset_sizes['train']}, Val: {dataset_sizes['val']}, Test: {dataset_sizes['test']}")
    
    # Check class distribution  
    class_counts = {phase: {'class_0': 0, 'class_1': 0, 'class_2': 0} for phase in ['train', 'val', 'test']}
    for phase in ['train', 'val', 'test']:
        for _, labels, _ in dataloaders[phase]:
            class_counts[phase]['class_0'] += (labels == 0).sum().item()
            class_counts[phase]['class_1'] += (labels == 1).sum().item()
            class_counts[phase]['class_2'] += (labels == 2).sum().item()
        print(f"{phase.capitalize()} set: Class 0: {class_counts[phase]['class_0']}, Class 1: {class_counts[phase]['class_1']}, Class 2: {class_counts[phase]['class_2']}")
    
    # Get initial time
    start_time = time.time()

    os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist
    csv_file = os.path.join(log_dir, f'training_log_{unique_id}.csv')

    # Initialize CSV file with headers
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Split', 'Loss', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Each epoch has training and validation phases
        for phase in ['train', 'val']:
            phase_start_time = time.time()
        
            if phase == 'train':
                model.train()  # Set model to training mode
                # print("Model set to training mode")
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            y_true = []
            y_pred = []
            batch_count = 0
            correct_count = 0
            total_count = 0
            
            # Display progress every N batches
            log_interval = max(1, len(dataloaders[phase]) // 10)
            
            # Iterate over data
            for batch_idx, (inputs, labels, _) in enumerate(dataloaders[phase]):
                batch_count += 1
                batch_size = inputs.size(0)
                inputs = inputs.squeeze(1).to(device)
                labels = labels.to(device)
                labels = labels.long()  # For CrossEntropyLoss

                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass - track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    inputs = inputs.squeeze(1)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                preds = torch.argmax(outputs, dim=1)
                correct_batch = (preds == labels).sum().item()
                correct_count += correct_batch
                total_count += batch_size
                
                # Update running loss
                running_loss += loss.item() * batch_size
                
                # Collect true labels and predictions for metrics
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                
                # Print progress
                if (batch_idx + 1) % log_interval == 0:
                    batch_acc = correct_batch / batch_size
                    avg_acc = correct_count / total_count
                    print(f"  Batch {batch_idx+1}/{len(dataloaders[phase])} - Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.4f}, Running Acc: {avg_acc:.4f}")
            
            # Calculate epoch loss and metrics
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_score(y_true, y_pred)
            epoch_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            epoch_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            epoch_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            # Log metrics to TensorBoard
            if summWriter != None:
                summWriter.add_scalar(f'{phase}/Loss', epoch_loss, epoch)
                summWriter.add_scalar(f'{phase}/Accuracy', epoch_acc, epoch)
                summWriter.add_scalar(f'{phase}/Precision', epoch_precision, epoch)
                summWriter.add_scalar(f'{phase}/Recall', epoch_recall, epoch)
                summWriter.add_scalar(f'{phase}/F1-Score', epoch_f1, epoch)

            # Calculate time taken for this phase
            phase_time = time.time() - phase_start_time
            
            # Log metrics
            print(f"\n{phase.capitalize()} statistics:")
            print(f"  Loss: {epoch_loss:.4f}")
            print(f"  Accuracy: {epoch_acc:.4f}")
            print(f"  Precision: {epoch_precision:.4f}")
            print(f"  Recall: {epoch_recall:.4f}")
            print(f"  F1-Score: {epoch_f1:.4f}")
            print(f"  Time: {phase_time:.2f} seconds")
            
            # Log metrics to CSV
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, phase, epoch_loss, epoch_acc, epoch_precision, epoch_recall, epoch_f1])
            
            # Update history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
                history['train_f1'].append(epoch_f1)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                history['val_f1'].append(epoch_f1)
                
                # Save the best model
                if epoch_acc > history['best_val_acc']:
                    # print(f"  New best model! Improved from {history['best_val_acc']:.4f} to {epoch_acc:.4f}")
                    history['best_val_acc'] = epoch_acc
                    history['best_epoch'] = epoch
                    torch.save(model.state_dict(), f'{checkpoint_dir}/best_model_{unique_id}.pth')
        
        # Calculate time taken for this epoch
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
    
    if summWriter != None:
        summWriter.close()  # Close the SummaryWriter

    
    return history



def evaluate_model(model, dataloader, criterion, device, log_dir=None, unique_id=None):
    """
    Evaluate the model on test data with detailed monitoring.
    """
    # print("\nStarting model evaluation on test set...")
    model.eval()
    
    running_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []  # Store prediction probabilities
    
    # print("Processing test batches:")
    batch_count = 0
    
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            batch_count += 1
            inputs = inputs.to(device)
            #labels = labels.to(device).float()
            labels = labels.to(device)
            labels = labels.long()  # For CrossEntropyLoss
            
            inputs = inputs.squeeze(1)
            outputs = model(inputs)
            # outputs = outputs.squeeze()  # Remove singleton dimension
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Convert outputs to binary predictions and probabilities
            preds = torch.argmax(outputs, dim=1)
            
            # Collect true labels, predictions, and probabilities for metrics
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    # Calculate metrics
    test_loss = running_loss / len(dataloader.dataset)
    # For multi-class classification with 3 classes
    test_acc = accuracy_score(y_true, y_pred)   
    test_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    test_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    test_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # If you want per-class metrics, you can use:
    class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Print all results in a clear format
    if unique_id == 'final_model':
        print("\nTest Results:")
        print("=" * 50)
        print(f"Loss:       {test_loss:.4f}")
        print(f"Accuracy:   {test_acc:.4f}")
        print(f"Precision:  {test_precision:.4f}")
        print(f"Recall:     {test_recall:.4f}")
        print(f"F1-Score:   {test_f1:.4f}")
        print(f"Confusion Matrix:")
        print(conf_matrix)
        print(f"Class Precision: {class_precision}")
        print(f"Class Recall: {class_recall}")
        print(f"Class F1: {class_f1}")
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1': test_f1,
        'confusion_matrix': conf_matrix
    }

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)	
    sampler = TPESampler(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    return sampler