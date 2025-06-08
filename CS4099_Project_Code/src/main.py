import os
import json
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, Dataset, ConcatDataset, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.cuda.amp as amp
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, balanced_accuracy_score, confusion_matrix
from densenet_model.densenet161 import initialize_densenet
from generate_graphs_results import generate_graphs
import config as config

# -------------------------------------------
# Logging Setup
# -------------------------------------------
logging.basicConfig(filename=config.log_filename,
                    level=logging.INFO, format='%(asctime)s - %(message)s')

# -------------------------------------------
# Dataset and Collate Functions
# -------------------------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            image, label = super(SafeImageFolder, self).__getitem__(index)
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        except (OSError, ValueError) as e:
            print(f"Warning: Skipping corrupted image at index {index}: {e}")
            return None

def safe_collate(batch):
    valid_batch = [item for item in batch if item is not None and isinstance(item, (tuple, list)) and len(item) == 2]
    if not valid_batch:
        return None
    return torch.utils.data.dataloader.default_collate(valid_batch)

# -------------------------------------------
# Probabilistic Augmentation Class
# -------------------------------------------
class ProbabilisticAugmentation:
    def __init__(self, base_transform, augmentation_transform, p=0.7):
        self.base_transform = base_transform
        self.augmentation_transform = augmentation_transform
        self.p = p

    def __call__(self, image):
        if random.random() < self.p:
            return self.augmentation_transform(image)
        else:
            return self.base_transform(image)

# -------------------------------------------
# Data Preparation Functions
# -------------------------------------------
def get_train_dataset(train_dir):
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    train_transform = ProbabilisticAugmentation(base_transform, augmentation_transform, p=0.5)
    dataset = SafeImageFolder(root=train_dir, transform=train_transform)
    dataset.samples = [s for s in dataset.samples if os.path.exists(s[0])]
    return dataset

def get_augmented_dataset(augment_dir):
    transform = transforms.Compose([
         transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    ])
    dataset = SafeImageFolder(root=augment_dir, transform=transform)
    dataset.samples = [s for s in dataset.samples if os.path.exists(s[0])]
    return dataset

def get_eval_loader(eval_dir, batch_size, num_workers):
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SafeImageFolder(root=eval_dir, transform=eval_transform)
    dataset.samples = [s for s in dataset.samples if os.path.exists(s[0])]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=safe_collate)
    return loader

def get_test_loader(test_dir, batch_size, num_workers):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SafeImageFolder(root=test_dir, transform=test_transform)
    dataset.samples = [s for s in dataset.samples if os.path.exists(s[0])]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=safe_collate)
    return loader

# -------------------------------------------
# Hard Negative Collection Dataset
# -------------------------------------------
class HardNegativesDataset(Dataset):
    def __init__(self, samples):
        """
        Args:
            samples (list): A list of tuples (image_tensor, label).
        """
        self.samples = [(img, torch.tensor(lbl, dtype=torch.long) if not torch.is_tensor(lbl) else lbl)
                        for img, lbl in samples]
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def get_calibration_loader(calibration_dir, batch_size=64, num_workers=4):
    # Use same transform as eval loader
    calib_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = SafeImageFolder(root=calibration_dir, transform=calib_transform)
    dataset.samples = [s for s in dataset.samples if os.path.exists(s[0])]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=safe_collate)
    return loader

def split_calibration(dataset, calibration_fraction=0.1):
    """
    Splits a dataset into training and calibration subsets.
    calibration_fraction: fraction (e.g., 0.1 for 10%) used for calibration.
    Returns: train_subset, calibration_subset
    """
    total_size = len(dataset)
    calib_size = int(total_size * calibration_fraction)
    train_size = total_size - calib_size
    return random_split(dataset, [train_size, calib_size])

# -------------------------------------------
# Calibration Function for Threshold Selection
# -------------------------------------------
def calibrate_threshold(model, calibration_loader, device, num_classes, candidate_thresholds=None):
    """
    Runs the model on the calibration set to find the best confidence threshold,
    based on maximum accuracy on accepted samples.
    """
    if candidate_thresholds is None:
        # candidate_thresholds = np.linspace(0.0, 0.90, 91)
        candidate_thresholds = np.arange(0.0, 0.96, 0.05)
        
    model.eval()
    confidences = []
    correctness = []
    
    with torch.no_grad():
        for data in calibration_loader:
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            sum_probs = probs.sum(dim=1)
            max_probs, predicted = probs.max(dim=1)
            c = 1.0 - sum_probs + max_probs  
            correct = (predicted == labels).long()
            confidences.extend(c.cpu().numpy())
            correctness.extend(correct.cpu().numpy())
    
    confidences = np.array(confidences)
    correctness = np.array(correctness)
    
    threshold_accuracies = {}
    best_threshold = 0.0
    best_accuracy = 0.0
    for tau in candidate_thresholds:
        accepted_mask = (confidences >= tau)
        if accepted_mask.sum() > 0:
            accuracy_accepted = correctness[accepted_mask].mean()
        else:
            accuracy_accepted = 0.0
        threshold_accuracies[round(tau,2)] = accuracy_accepted
        if accuracy_accepted > best_accuracy:
            best_accuracy = accuracy_accepted
            best_threshold = tau
    print(f"[Calibration] Best threshold = {best_threshold:.3f} (Accuracy on accepted = {best_accuracy:.3f})")
    return best_threshold, threshold_accuracies

# -------------------------------------------
# Evaluation Function with Custom Confidence Measure
# -------------------------------------------
def evaluate_model(model, data_loader, device, num_classes, confidence_threshold, train):
    model.eval()
    overall_labels = []   # All predictions
    overall_predictions = []
    accepted_labels = []  # Only samples where custom confidence >= threshold
    accepted_predictions = []
    hard_negatives = []   # (optional: for training)
    total_count = 0
    rejected_count = 0
    MAX_HARD_NEGATIVES = 3500

    with torch.no_grad():
        for data in data_loader:
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            sum_probs = probs.sum(dim=1)
            max_probs, predicted = probs.max(dim=1)
            c = 1.0 - sum_probs + max_probs   # custom confidence measure
            batch_size = inputs.size(0)
            total_count += batch_size

            overall_labels.extend(labels.cpu().numpy().tolist())
            overall_predictions.extend(predicted.cpu().numpy().tolist())

            accepted_mask = (c >= confidence_threshold)
            accepted_labels.extend(labels[accepted_mask].cpu().numpy().tolist())
            accepted_predictions.extend(predicted[accepted_mask].cpu().numpy().tolist())

            if train:
                rejected_mask = ~accepted_mask
                num_rejected = rejected_mask.sum().item()
                rejected_count += num_rejected

                # Only collect new negatives if we haven't reached the max
                if len(hard_negatives) < MAX_HARD_NEGATIVES:
                    new_negatives = list(zip(inputs[rejected_mask].cpu(), labels[rejected_mask].cpu()))
                    remaining_capacity = MAX_HARD_NEGATIVES - len(hard_negatives)
                    if len(new_negatives) > remaining_capacity:
                        new_negatives = new_negatives[:remaining_capacity]
                    hard_negatives.extend(new_negatives)

    overall_accuracy = accuracy_score(overall_labels, overall_predictions)
    overall_f1 = f1_score(overall_labels, overall_predictions, average='weighted', zero_division=1)
    overall_recall = recall_score(overall_labels, overall_predictions, average='weighted', zero_division=1)
    overall_precision = precision_score(overall_labels, overall_predictions, average='weighted', zero_division=1)
    overall_balanced = balanced_accuracy_score(overall_labels, overall_predictions)
    overall_conf_matrix = confusion_matrix(overall_labels, overall_predictions, normalize='true')

    if len(accepted_labels) > 0:
        accepted_accuracy = accuracy_score(accepted_labels, accepted_predictions)
        accepted_f1 = f1_score(accepted_labels, accepted_predictions, average='weighted', zero_division=1)
        accepted_recall = recall_score(accepted_labels, accepted_predictions, average='weighted', zero_division=1)
        accepted_precision = precision_score(accepted_labels, accepted_predictions, average='weighted', zero_division=1)
        accepted_balanced = balanced_accuracy_score(accepted_labels, accepted_predictions)
        accepted_conf_matrix = confusion_matrix(accepted_labels, accepted_predictions, normalize='true')
    else:
        accepted_accuracy = accepted_f1 = accepted_recall = accepted_precision = accepted_balanced = None
        accepted_conf_matrix = None

    metrics = {
        'overall': {
            'accuracy': overall_accuracy,
            'balanced_accuracy': overall_balanced,
            'f1_score': overall_f1,
            'recall': overall_recall,
            'precision': overall_precision,
            'confusion_matrix': overall_conf_matrix.tolist() if overall_conf_matrix is not None else None,
        },
        'accepted': {
            'accuracy': accepted_accuracy,
            'balanced_accuracy': accepted_balanced,
            'f1_score': accepted_f1,
            'recall': accepted_recall,
            'precision': accepted_precision,
            'confusion_matrix': accepted_conf_matrix.tolist() if accepted_conf_matrix is not None else None,
        },
        'rejection_rate': rejected_count / total_count if total_count > 0 else 0.0,
        'num_rejected': rejected_count,
        'num_total': total_count
    }
    return metrics, hard_negatives, total_count, rejected_count

def evaluate_overall_accuracy(model, data_loader, device):
    model.eval()
    overall_labels = []   # All ground truth labels
    overall_predictions = []  # All predictions

    with torch.no_grad():
        for data in data_loader:
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # You can either use argmax directly or compute softmax then argmax
            predicted = torch.argmax(outputs, dim=1)
            overall_labels.extend(labels.cpu().numpy().tolist())
            overall_predictions.extend(predicted.cpu().numpy().tolist())

    overall_accuracy = accuracy_score(overall_labels, overall_predictions)
    return overall_accuracy

def evaluate_training_accuracy(model, train_loader, device):
    model.eval()  # Make sure the model is in evaluation mode
    train_labels = []
    train_predictions = []

    with torch.no_grad():
        for data in train_loader:
            if data is None:
                continue
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            train_labels.extend(labels.cpu().numpy().tolist())
            train_predictions.extend(predicted.cpu().numpy().tolist())

    training_accuracy = accuracy_score(train_labels, train_predictions)
    return training_accuracy

# -------------------------------------------
# Training Loop with Calibration and Final Test Evaluation
# -------------------------------------------
def train_model(train_dir, eval_dir, test_dir, augment_dir,
                num_epochs, batch_size, learning_rate, num_workers,
                initial_threshold, patience, calib_fraction=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data (original and augmented)
    original_train_dataset = get_train_dataset(train_dir)
    augmented_dataset = get_augmented_dataset(augment_dir)
    combined_dataset = ConcatDataset([original_train_dataset, augmented_dataset])
    
    # Split the combined dataset into training and calibration subsets
    train_subset, calibration_subset = split_calibration(combined_dataset, calibration_fraction=calib_fraction)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    calibration_loader = DataLoader(calibration_subset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True, collate_fn=safe_collate)
    
    # Load evaluation and test data
    eval_loader = get_eval_loader(eval_dir, batch_size, num_workers)
    test_loader = get_test_loader(test_dir, batch_size, num_workers)

    hard_negatives = []
    model = initialize_densenet(num_classes=21)
    model.to(device)
    criterion = nn.CrossEntropyLoss()


    if config.updated_baseline: # For Updated Baseline
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9,0.999), weight_decay=0.001)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60], gamma=0.1)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9,0.999))
    
    scaler = torch.amp.GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')

    experiment_results = {}
    best_eval_accuracy = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):

        if hard_negatives:
            hard_neg_dataset = HardNegativesDataset(hard_negatives)
            train_dataset = ConcatDataset([combined_dataset, hard_neg_dataset])
        else:
            train_dataset = combined_dataset

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
    
        if config.updated_baseline:
            scheduler.step()

        if config.hard_negative_mining==True & epoch % 3 == 0:
            
            train_metrics, hard_negatives, total_count, rejected_count = evaluate_model(model, train_loader, device, num_classes=21,
                                                    confidence_threshold=initial_threshold, train=True)
            eval_metrics, _, _, _ = evaluate_model(model, eval_loader, device, num_classes=21,
                                                confidence_threshold=initial_threshold, train=False)
            
            rejection_rate = rejected_count / total_count if total_count > 0 else 0.0
            logging.info(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")
            logging.info(f" Training Metrics (Threshold={initial_threshold}): {train_metrics}")
            logging.info(f" Eval Metrics (Threshold={initial_threshold}): {eval_metrics}")
            
            experiment_results[epoch+1] = {
                'train_loss': avg_loss,
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'rejection_rate': rejection_rate,
                'num_rejected': rejected_count,
            }

            current_eval_acc = eval_metrics['overall'].get('accuracy', 0)
            if current_eval_acc > best_eval_accuracy:
                best_eval_accuracy = current_eval_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logging.info(f"No improvement for {early_stopping_counter} epoch(s).")
                if early_stopping_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}.")
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
        else :
            logging.info(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")



        if config.hard_negative_mining==False & epoch % 3 == 0:

            overall_eval_acc = evaluate_overall_accuracy(model, eval_loader, device)
            overall_train_acc = evaluate_training_accuracy(model, train_loader, device)

            current_eval_acc = overall_eval_acc
            if current_eval_acc > best_eval_accuracy:
                best_eval_accuracy = current_eval_acc
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                logging.info(f"No improvement for {early_stopping_counter} epoch(s).")
                if early_stopping_counter >= patience:
                    logging.info(f"Early stopping triggered at epoch {epoch+1}.")
                    print(f"Early stopping triggered at epoch {epoch+1}.")
                    break
            
            logging.info(f"Eval Accuracy: {overall_eval_acc:.4f}")
            logging.info(f"Train Accuracy: {overall_train_acc:.4f}")

            experiment_results[epoch+1] = {
                'train_loss': avg_loss,
                'overall_eval_accuracy': overall_eval_acc,
                'overall_train_accuracy': overall_train_acc,
            }

    with open(config.output_json_file, 'w') as f:
        json.dump(experiment_results, f, indent=4)

    # --- Calibration Step ---
    print("Calibrating confidence threshold using calibration set...")
    best_threshold, threshold_accuracies = calibrate_threshold(model, calibration_loader, device, num_classes=21)
    print(f"Calibrated threshold = {best_threshold:.3f}")

    final_validation_metrics, _, total_count, rejected_count = evaluate_model(
        model, eval_loader, device, num_classes=21,
        confidence_threshold=best_threshold, train=False
    )

    logging.info("Final Validation Evaluation:")
    logging.info(f"Calibrated Threshold = {best_threshold:.3f}")
    logging.info(f"Final Validation Metrics: {final_validation_metrics}")
    logging.info(f"Rejection Rate: {rejected_count / total_count:.3f} ({rejected_count} / {total_count})")
    logging.info(f"Rejection Count: {rejected_count}")

    if config.run_mode == "test":
        # --- Test Evaluation after convergence ---
        final_test_metrics, _, total_count, rejected_count = evaluate_model(
            model, test_loader, device, num_classes=21,
            confidence_threshold=best_threshold, train=False
        )

        logging.info("Calibration and Final Test Evaluation:")
        logging.info(f"Calibrated Threshold = {best_threshold:.3f}")
        logging.info(f"Final Test Metrics: {final_test_metrics}")
        logging.info(f"Rejection Rate: {rejected_count / total_count:.3f} ({rejected_count} / {total_count})")
        logging.info(f"Rejection Count: {rejected_count}")

        extra_metrics = {
            'final_validation_metrics': final_validation_metrics,
            'final_test_metrics': final_test_metrics,
            'threshold_accuracies': threshold_accuracies,
            'best_threshold': best_threshold
        }
    else :
        # --- Validation Evaluation after convergence ---
        logging.info("Calibration and Final Test Evaluation:")
        logging.info(f"Calibrated Threshold = {best_threshold:.3f}")
        logging.info(f"Rejection Rate: {rejected_count / total_count:.3f} ({rejected_count} / {total_count})")
        logging.info(f"Rejection Count: {rejected_count}")

        extra_metrics = {
            'final_validation_metrics': final_validation_metrics,
            'threshold_accuracies': threshold_accuracies,
            'best_threshold': best_threshold
        }

    with open(config.output_json_file_extra, 'w') as f:
        json.dump(extra_metrics, f, indent=4)

     # Save the final model 
    torch.save(model.state_dict(),config.saved_model_path)


# -------------------------------------------
# Utility: Split Calibration Set from Dataset
# -------------------------------------------
def split_calibration(dataset, calibration_fraction=0.1):
    total_size = len(dataset)
    calib_size = int(total_size * calibration_fraction)
    train_size = total_size - calib_size
    return random_split(dataset, [train_size, calib_size])

# -------------------------------------------
# Main Execution
# -------------------------------------------
if __name__ == '__main__':
    train_dir = config.train_dir
    eval_dir = config.eval_dir
    test_dir = config.test_dir

    if config.updated_baseline:
        augment_dir = config.augment_dir_updated
    else:
        augment_dir = config.augment_dir

    num_epochs = config.num_epochs
    batch_size = config.batch_size
    learning_rate = config.learning_rate
    num_workers = config.num_workers
    initial_threshold = config.confidence_threshold
    patience = config.patience

    os.makedirs(config.output_path, exist_ok=True)

    train_model(train_dir, eval_dir,
                test_dir, augment_dir, num_epochs, batch_size, learning_rate, num_workers,
                initial_threshold, patience)
    

    generate_graphs(config.output_json_file, config.output_json_file_extra, config.output_path)
    
