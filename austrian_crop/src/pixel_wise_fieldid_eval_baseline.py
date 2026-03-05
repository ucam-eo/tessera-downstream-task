import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from collections import Counter
import logging
from datetime import datetime
import time

# -----------------------------------------------------------------------------
# Configuration Parameters
# -----------------------------------------------------------------------------

# Random seeds for reproducibility
NUMPY_SEED = 42
TORCH_SEED = 42

# Data file paths
BANDS_FILE_PATH = "data/bands_downsample_100.npy"
LABEL_FILE_PATH = "data/fieldtype_17classes_downsample_100.npy"
FIELD_ID_FILE_PATH = "data/fieldid_downsample_100.npy"
UPDATED_FIELDDATA_PATH = 'data/updated_fielddata.csv'
SAR_ASC_BANDS_FILE_PATH = "data/sar_ascending_downsample_100.npy"
SAR_DESC_BANDS_FILE_PATH = "data/sar_descending_downsample_100.npy"

# Log settings
LOG_FILE = "feature_analysis.log"

# Dataset split ratios
TRAINING_RATIO = 0.3
VAL_TEST_SPLIT_RATIO = 1/7.0  # Validation to test set split ratio

# Model selection
MODEL = "MLP"  # Options: "LogisticRegression", "RandomForest", or "MLP"

# Training hyperparameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
PATIENCE = 5  # Early stopping parameter
WEIGHT_DECAY = 0.01

# Processing parameters
NJOBS = 12
CHUNK_SIZE = 1000

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sentinel-2 normalization parameters
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                        2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                       1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)

# Sentinel-1 normalization parameters
S1_BAND_MEAN = np.array([5484.0407, 3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334, 1726.0670], dtype=np.float32)

# Class names for reference
CLASS_NAMES = [
    "Legume", "Soy", "Summer Grain", "Winter Grain", "Corn", "Sunflower",
    "Mustard", "Potato", "Beet", "Squash", "Grapes", "Tree Fruit",
    "Cover Crop", "Grass", "Fallow", "Other (Plants)", "Other (Non Plants)"
]

# -----------------------------------------------------------------------------
# MLP Model Definition
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(MLP, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.output = nn.Linear(hidden_sizes[2], num_classes)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.output(x)
        return x

# -----------------------------------------------------------------------------
# Model Training Functions
# -----------------------------------------------------------------------------
def train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size):
    """Train MLP model and return the trained model"""
    logging.info(f"Starting MLP training with input size: {input_size}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    hidden_sizes = [512, 256, 128]
    model = MLP(input_size, hidden_sizes, num_classes).to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    logging.info(f"MLP model architecture:\n{model}")
    
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        scheduler.step(val_loss)
        
        logging.info(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            logging.info(f"Saving best model with validation loss: {best_val_loss:.4f}")
        else:
            early_stop_counter += 1
        
        # Early stopping
        if early_stop_counter >= PATIENCE:
            logging.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def mlp_predict(model, X):
    """Make predictions using MLP model"""
    model.eval()
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    
    # Batch predict for large datasets
    batch_size = 1024
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch_X = X_tensor[i:i+batch_size]
            outputs = model(batch_X)
            _, preds = torch.max(outputs, 1)
            predictions.append(preds.cpu().numpy())
    
    return np.concatenate(predictions)

class MLPWrapper:
    def __init__(self, model):
        self.model = model
        
    def predict(self, X):
        return mlp_predict(self.model, X)

# -----------------------------------------------------------------------------
# Data Processing Functions
# -----------------------------------------------------------------------------
def process_chunk(h_start, h_end, w_start, w_end, file_path, valid_classes, train_fids, val_fids, test_fids):
    """Process a chunk of data and return train/val/test splits"""
    logging.info(f"Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
    
    # Load and normalize S2 data
    tile_chunk = np.load(file_path)[:, h_start:h_end, w_start:w_end, :]
    tile_chunk = (tile_chunk - S2_BAND_MEAN) / S2_BAND_STD
    time_steps, h, w, bands = tile_chunk.shape
    s2_band_chunk = tile_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands)
    
    # Load and normalize S1 data
    sar_asc_chunk = np.load(SAR_ASC_BANDS_FILE_PATH)[:, h_start:h_end, w_start:w_end]
    sar_desc_chunk = np.load(SAR_DESC_BANDS_FILE_PATH)[:, h_start:h_end, w_start:w_end]
    sar_chunk = np.concatenate((sar_asc_chunk, sar_desc_chunk), axis=0)
    sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD
    time_steps, h, w, bands = sar_chunk.shape
    sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands)
    
    # Concatenate S2 and S1 features
    X_chunk = np.concatenate((s2_band_chunk, sar_band_chunk), axis=1)
    
    # Load labels and field IDs
    labels = np.load(LABEL_FILE_PATH).astype(np.int64)
    field_ids = np.load(FIELD_ID_FILE_PATH)
    
    y_chunk = labels[h_start:h_end, w_start:w_end].ravel()
    fieldid_chunk = field_ids[h_start:h_end, w_start:w_end].ravel()
    
    # Filter valid data
    valid_mask = np.isin(y_chunk, list(valid_classes))
    X_chunk, y_chunk, fieldid_chunk = X_chunk[valid_mask], y_chunk[valid_mask], fieldid_chunk[valid_mask]
    
    # Split into train/val/test sets based on field_id
    train_mask = np.isin(fieldid_chunk, train_fids)
    val_mask = np.isin(fieldid_chunk, val_fids)
    test_mask = np.isin(fieldid_chunk, test_fids)
    
    return (X_chunk[train_mask], y_chunk[train_mask], 
            X_chunk[val_mask], y_chunk[val_mask],
            X_chunk[test_mask], y_chunk[test_mask])

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    # Fix random seeds for reproducibility
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
    )
    
    timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
    logging.info("Program started.")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Training ratio: {TRAINING_RATIO}")
    logging.info(f"Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
    logging.info(f"Selected model: {MODEL}")
    
    # -------------------------
    # Load labels and field IDs
    # -------------------------
    logging.info("Loading labels and field IDs...")
    labels = np.load(LABEL_FILE_PATH).astype(np.int64)
    field_ids = np.load(FIELD_ID_FILE_PATH)
    H, W = labels.shape
    logging.info(f"Data dimensions: {H}x{W}")
    
    # Select valid classes
    logging.info("Identifying valid classes...")
    class_counts = Counter(labels.ravel())
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
    valid_classes.discard(0)
    logging.info(f"Valid classes: {sorted(valid_classes)}")
    
    # -------------------------
    # Train/validation/test set split
    # -------------------------
    logging.info("Splitting data into train/val/test sets...")
    fielddata_df = pd.read_csv(UPDATED_FIELDDATA_PATH)
    area_summary = fielddata_df.groupby('SNAR_CODE')['area_m2'].sum().reset_index()
    area_summary.rename(columns={'area_m2': 'total_area'}, inplace=True)
    
    # Collect training set field IDs
    train_fids = []
    for _, row in area_summary.iterrows():
        sn_code = row['SNAR_CODE']
        total_area = row['total_area']
        target_area = total_area * TRAINING_RATIO
        rows_sncode = fielddata_df[fielddata_df['SNAR_CODE'] == sn_code].sort_values(by='area_m2')
        selected_fids = []
        selected_area_sum = 0
        for _, r2 in rows_sncode.iterrows():
            if selected_area_sum < target_area:
                selected_fids.append(int(r2['fid_1']))
                selected_area_sum += r2['area_m2']
            else:
                break
        train_fids.extend(selected_fids)
    
    train_fids = list(set(train_fids))
    logging.info(f"Number of selected train field IDs: {len(train_fids)}")
    
    # Split remaining field IDs into validation and test sets
    all_fields = fielddata_df['fid_1'].unique().astype(int)
    set_train = set(train_fids)
    set_all = set(all_fields)
    remaining = list(set_all - set_train)
    remaining = np.array(remaining)
    np.random.shuffle(remaining)
    val_count = int(len(remaining) * VAL_TEST_SPLIT_RATIO)
    val_fids = remaining[:val_count]
    test_fids = remaining[val_count:]
    train_fids = np.array(train_fids)
    logging.info(f"Train fields: {len(train_fids)}, Val fields: {len(val_fids)}, Test fields: {len(test_fids)}")
    
    # -------------------------
    # Chunk processing
    # -------------------------
    chunks = [(h, min(h+CHUNK_SIZE, H), w, min(w+CHUNK_SIZE, W))
             for h in range(0, H, CHUNK_SIZE)
             for w in range(0, W, CHUNK_SIZE)]
    logging.info(f"Total chunks: {len(chunks)}")
    
    results = Parallel(n_jobs=NJOBS)(
        delayed(process_chunk)(h_start, h_end, w_start, w_end, BANDS_FILE_PATH, valid_classes, train_fids, val_fids, test_fids)
        for h_start, h_end, w_start, w_end in chunks
    )
    
    # Combine results
    X_train = np.vstack([res[0] for res in results if res[0].size > 0])
    y_train = np.hstack([res[1] for res in results if res[1].size > 0])
    X_val = np.vstack([res[2] for res in results if res[2].size > 0])
    y_val = np.hstack([res[3] for res in results if res[3].size > 0])
    X_test = np.vstack([res[4] for res in results if res[4].size > 0])
    y_test = np.hstack([res[5] for res in results if res[5].size > 0])
    
    logging.info(f"Data split summary:")
    logging.info(f"  Train set: {X_train.shape[0]} samples")
    logging.info(f"  Validation set: {X_val.shape[0]} samples")
    logging.info(f"  Test set: {X_test.shape[0]} samples")
    logging.info(f"X_train shape: {X_train.shape}")
    
    input_size = X_train.shape[1]
    logging.info(f"Input feature dimension: {input_size}")
    
    # -------------------------
    # Model training
    # -------------------------
    logging.info(f"\nTraining {MODEL}...")
    
    if MODEL == "LogisticRegression":
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=1e4,
            max_iter=100000,
            n_jobs=NJOBS,
            random_state=NUMPY_SEED
        )
        model.fit(X_train, y_train)
        
    elif MODEL == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=NJOBS,
            random_state=NUMPY_SEED
        )
        model.fit(X_train, y_train)
        
    elif MODEL == "MLP":
        num_classes = max(valid_classes) + 1
        logging.info(f"Number of classes: {num_classes}")
        
        mlp_model = train_mlp(X_train, y_train, X_val, y_val, num_classes, input_size)
        model = MLPWrapper(mlp_model)
        
    else:
        raise ValueError(f"Unknown model type: {MODEL}. Use 'LogisticRegression', 'RandomForest', or 'MLP'")
    
    # -------------------------
    # Evaluation
    # -------------------------
    logging.info("Evaluating model on test set...")
    y_pred = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average='weighted')
    
    logging.info(f"Test Accuracy: {test_accuracy:.4f}")
    logging.info(f"Test F1 Score: {test_f1:.4f}")
    logging.info("Classification Report (Test Set):\n" + classification_report(y_test, y_pred, digits=4))
    
    # Per-class accuracy statistics
    class_accuracies = {}
    for cls in sorted(valid_classes):
        cls_mask = y_test == cls
        if np.sum(cls_mask) > 0:
            cls_accuracy = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
            class_accuracies[cls] = cls_accuracy * 100
            class_name = CLASS_NAMES[cls-1] if cls-1 < len(CLASS_NAMES) else f"Class {cls}"
            logging.info(f"Class {cls} ({class_name}) accuracy: {cls_accuracy*100:.2f}%")
    
    logging.info("\nAnalysis completed!")
    logging.info(f"Training completed. Final Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}")

if __name__ == "__main__":
    main()