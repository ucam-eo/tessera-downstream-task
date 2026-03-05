import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import logging
from einops import rearrange
import pandas as pd
from datetime import datetime

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------

# Random seeds for reproducibility
NUMPY_SEED = 42
TORCH_SEED = 42

# Embedding file paths

# TESSERA
REPRESENTATION_FILE_PATH = "data/austrian_crop_tessera_embedding_downsample_100.npy"
# Google Satellite Embedding
# REPRESENTATION_FILE_PATH = "data/austrian_crop_gse_embedding_downsample_100.npy"
# PRESTO
# REPRESENTATION_FILE_PATH = "data/austrian_crop_presto_embeddings_downsample_100.npy"

LABEL_FILE_PATH = "data/fieldtype_17classes_downsample_100.npy"
FIELD_ID_FILE_PATH = "data/fieldid_downsample_100.npy"
UPDATED_FIELDDATA_PATH = "data/updated_fielddata.csv"

# Dataset split ratios
TRAINING_RATIO = 0.25  # Proportion of all samples for training set
VAL_TEST_SPLIT_RATIO = 1/7.0  # In remaining samples, validation to test ratio is 1:7

# Training hyperparameters
BATCH_SIZE = 8192
NUM_EPOCHS = 200
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
LOG_INTERVAL = 32  # Print logs every N steps

# Model architecture parameters
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 256
DROPOUT_RATE = 0.3

# Checkpoint settings
CHECKPOINT_SAVE_FOLDER = "checkpoints/downstream/"

# -----------------------------------------------------------------------------
# Dataset class: supports train, validation, and test splits
# -----------------------------------------------------------------------------
class LandClassificationDataset(Dataset):
    def __init__(self, representations, labels, field_ids, split='train',
                 train_field_ids=None, val_field_ids=None, test_field_ids=None):
        """
        Args:
            representations (ndarray): Feature representations with shape (H, W, D)
            labels (ndarray): Labels with shape (H, W)
            field_ids (ndarray): Field IDs with shape (H, W)
            split (str): 'train', 'val' or 'test'
            train_field_ids (list): List of field IDs for training set
            val_field_ids (list): List of field IDs for validation set
            test_field_ids (list): List of field IDs for test set
        """
        valid_mask = labels != 0  # Filter out invalid pixels with label 0
        if split == 'train':
            assert train_field_ids is not None, "Training set requires train_field_ids"
            mask = np.isin(field_ids, train_field_ids) & valid_mask
        elif split == 'val':
            assert val_field_ids is not None, "Validation set requires val_field_ids"
            mask = np.isin(field_ids, val_field_ids) & valid_mask
        elif split == 'test':
            assert test_field_ids is not None, "Test set requires test_field_ids"
            mask = np.isin(field_ids, test_field_ids) & valid_mask
        else:
            raise ValueError("split must be 'train', 'val' or 'test'")
        
        # Extract pixels that meet conditions, shape becomes (N, D)
        self.representations = rearrange(representations[mask], 'n d -> n d')
        self.labels = labels[mask]

    def __len__(self):
        return len(self.representations)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.representations[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -----------------------------------------------------------------------------
# Classification head network definition
# -----------------------------------------------------------------------------
class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, HIDDEN_DIM_1)
        self.relu1 = nn.ReLU()
        self.ln1 = nn.BatchNorm1d(HIDDEN_DIM_1)
        self.dropout1 = nn.Dropout(DROPOUT_RATE)

        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.relu2 = nn.ReLU()
        self.ln2 = nn.BatchNorm1d(HIDDEN_DIM_2)
        self.dropout2 = nn.Dropout(DROPOUT_RATE)

        self.fc3 = nn.Linear(HIDDEN_DIM_2, num_classes)

    def forward(self, x):
        x = self.relu1(self.ln1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.relu2(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def load_and_dequantize_representation(representation_file_path, scales_file_path):
    """
    Load and dequantize int8 representations back to float32.
    
    Args:
        representation_file_path: Path to the int8 representation file (H,W,C)
        scales_file_path: Path to the float32 scales file (H,W)
    
    Returns:
        representation_f32: float32 ndarray of shape (H,W,C)
    """
    # Load the files
    representation_int8 = np.load(representation_file_path)  # (H, W, C), dtype=int8
    scales = np.load(scales_file_path)  # (H, W), dtype=float32
    
    # Convert int8 to float32 for computation
    representation_f32 = representation_int8.astype(np.float32)
    
    # Expand scales to match representation shape
    # scales shape: (H, W) -> (H, W, 1)
    scales_expanded = scales[..., np.newaxis]
    
    # Dequantize by multiplying with scales
    representation_f32 = representation_f32 * scales_expanded
    
    return representation_f32

# -----------------------------------------------------------------------------
# Main function: data loading, dataset splitting, training, validation and final testing
# -----------------------------------------------------------------------------
def main():
    # Fix random seeds for reproducibility
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(TORCH_SEED)
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    
    # -------------------------
    # Load representations, labels and field IDs
    # -------------------------
    representations = np.load(REPRESENTATION_FILE_PATH)  # (H, W, D)
    labels = np.load(LABEL_FILE_PATH).astype(np.int64)     # (H, W)
    field_ids = np.load(FIELD_ID_FILE_PATH).astype(np.int64) # (H, W)
    
    # Check for NaN values in representations and replace with 0
    nan_mask = np.isnan(representations)
    nan_count = np.sum(nan_mask)
    if nan_count > 0:
        logging.info(f"Found {nan_count} NaN values in representations ({nan_count/representations.size*100:.2f}% of total values)")
        logging.info(f"Replacing NaN values with 0...")
        representations[nan_mask] = 0.0
        logging.info("NaN replacement completed.")
    else:
        logging.info("No NaN values found in representations.")

    # Check and adjust representation H and W to match labels
    if representations.shape[0] != labels.shape[0] or representations.shape[1] != labels.shape[1]:
        logging.info(f"Representation shape {representations.shape[:2]} does not match labels shape {labels.shape}. Resizing representation...")
        # Convert (H, W, D) to (D, H, W) for interpolation
        representations_tensor = torch.tensor(representations).permute(2, 0, 1).unsqueeze(0)  # (1, D, H, W)
        # Use bilinear interpolation for resizing
        resized_representations_tensor = nn.functional.interpolate(
            representations_tensor,
            size=(labels.shape[0], labels.shape[1]),
            mode='bilinear',
            align_corners=False
        )
        # Convert (1, D, H_new, W_new) back to (H_new, W_new, D) and convert to numpy
        representations = resized_representations_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        logging.info(f"Resized representation shape: {representations.shape}")
    
    # Remap labels: ensure labels are continuous (old code treated label 0 as invalid)
    unique_labels = np.unique(labels)
    print(f"Unique labels: {unique_labels}")
    num_classes = len(unique_labels)
    print(f"Number of valid classes: {num_classes}")
    label_map = {label: i for i, label in enumerate(unique_labels)}
    labels = np.vectorize(label_map.get)(labels)
    
    # -------------------------
    # Use updated_fielddata.csv to split train, validation and test sets
    # Old code groups by SNAR_CODE, takes 30% of area from each group as training set,
    # randomly splits remaining into validation and test sets
    # -------------------------
    fielddata_df = pd.read_csv(UPDATED_FIELDDATA_PATH)
    area_summary = fielddata_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
    area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)
    
    all_selected_fids = []
    for _, row in area_summary.iterrows():
        sn_code = row['SNAR_CODE']
        total_area = row['total_area']
        target_area = total_area * TRAINING_RATIO
        selected_rows = fielddata_df[fielddata_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
        
        selected_fids = []
        selected_area_sum = 0
        for _, selected_row in selected_rows.iterrows():
            if selected_area_sum < target_area:
                selected_fids.append(int(selected_row['fid_1']))
                selected_area_sum += selected_row['area_m2']
            else:
                break
        all_selected_fids.extend(selected_fids)
    
    all_selected_fids = [int(fid) for fid in all_selected_fids]
    logging.info(f"Selected field IDs for training (last 20): {all_selected_fids[-20:]}")
    
    # Split validation and test sets: randomly split remaining fields with 1/7 for validation, rest for test
    all_fields = fielddata_df['fid_1'].unique()
    set_all = set(all_fields)
    set_train = set(all_selected_fids)
    remaining = list(set_all - set_train)
    np.random.shuffle(remaining)
    val_count = int(len(remaining) * VAL_TEST_SPLIT_RATIO)
    val_fids = remaining[:val_count]
    test_fids = remaining[val_count:]
    
    # -------------------------
    # Construct datasets and DataLoaders
    # -------------------------
    train_dataset = LandClassificationDataset(representations, labels, field_ids, split='train', train_field_ids=all_selected_fids)
    val_dataset = LandClassificationDataset(representations, labels, field_ids, split='val', val_field_ids=val_fids)
    test_dataset = LandClassificationDataset(representations, labels, field_ids, split='test', test_field_ids=test_fids)
    
    logging.info(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    # -------------------------
    # Model, loss function, optimizer
    # -------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = representations.shape[-1]
    model = ClassificationHead(input_dim, num_classes).to(device)
    # Use cross entropy
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # -------------------------
    # Training configuration and checkpoint setup
    # -------------------------
    best_val_f1 = 0.0
    best_val_accuracy = 0.0
    best_epoch = 0
    os.makedirs(CHECKPOINT_SAVE_FOLDER, exist_ok=True)
    best_checkpoint_name = f"austrian_crop_representation_checkpoint_{timestamp}_best.pt"
    best_checkpoint_path = os.path.join(CHECKPOINT_SAVE_FOLDER, best_checkpoint_name)
    
    # -------------------------
    # Training and validation loop
    # -------------------------
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        for step, (reps, labels_batch) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{NUM_EPOCHS}")):
            reps, labels_batch = reps.to(device), labels_batch.to(device)
            optimizer.zero_grad()
            outputs = model(reps)
            # Reshape output to [N, num_classes], same for labels
            outputs = outputs.view(-1, num_classes)
            labels_batch = labels_batch.view(-1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            train_targets.extend(labels_batch.cpu().numpy())
            
            if (step + 1) % LOG_INTERVAL == 0:
                if len(train_preds) > 0:
                    flat_preds = np.array(train_preds)
                    flat_targets = np.array(train_targets)
                    # Ignore invalid samples with label 0
                    valid_mask = flat_targets != 0
                    valid_preds = flat_preds[valid_mask]
                    valid_targets = flat_targets[valid_mask]
                    
                    train_loss = running_loss / LOG_INTERVAL
                    train_accuracy = accuracy_score(valid_targets, valid_preds)
                    train_f1 = f1_score(valid_targets, valid_preds, average='weighted')
                    print(f"Step [{step+1}/{len(train_loader)}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                    logging.info(f"Step [{step+1}/{len(train_loader)}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    logging.info(f"Step [{step+1}/{len(train_loader)}] - No valid predictions to log.")
                running_loss = 0.0
                train_preds, train_targets = [], []
                
        # Perform validation after each epoch
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for reps, labels_batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{NUM_EPOCHS}"):
                reps, labels_batch = reps.to(device), labels_batch.to(device)
                outputs = model(reps)
                val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                val_targets.extend(labels_batch.cpu().numpy())
        val_accuracy = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Validation Accuracy: {val_accuracy:.4f}, Validation F1: {val_f1:.4f}")
        
        # Save checkpoint if current epoch has better validation performance
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_checkpoint_path)
            logging.info(f"Best checkpoint saved at epoch {best_epoch} with val_accuracy: {best_val_accuracy:.4f}, val_f1: {best_val_f1:.4f}")
            # Print validation set classification report
            class_report = classification_report(val_targets, val_preds, digits=4)
            print("\nValidation Classification Report:\n")
            print(class_report)
            logging.info("\nValidation Classification Report:\n" + class_report)
    
    print(f"Training completed. Best Validation Accuracy: {best_val_accuracy:.4f}, Best Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    logging.info(f"Training completed. Best Validation Accuracy: {best_val_accuracy:.4f}, Best Validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
    
    # -------------------------
    # Final evaluation on test set: load best checkpoint
    # -------------------------
    logging.info(f"Loading best checkpoint from epoch {best_epoch} for final test evaluation.")
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for reps, labels_batch in tqdm(test_loader, desc="Final Test Evaluation"):
            reps, labels_batch = reps.to(device), labels_batch.to(device)
            outputs = model(reps)
            test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            test_targets.extend(labels_batch.cpu().numpy())
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds, average='weighted')
    test_class_report = classification_report(test_targets, test_preds, digits=4)
    
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print(f"Final Test F1: {test_f1:.4f}")
    print("\nTest Classification Report:\n")
    print(test_class_report)
    logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Final Test F1: {test_f1:.4f}")
    logging.info("\nTest Classification Report:\n" + test_class_report)

if __name__ == "__main__":
    main()