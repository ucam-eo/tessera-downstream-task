import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from joblib import Parallel, delayed
from collections import Counter
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import time
import os
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ----------------- Command Line Argument Parsing -----------------
parser = argparse.ArgumentParser(description='Land Classification with Multiple Experiments')
parser.add_argument('--sample_per_pixel', type=int, default=10, help='Number of samples per class (default: 10)')
parser.add_argument('--num_experiment', type=int, default=200, help='Number of experiments to run (default: 200)')
parser.add_argument('--result_dir', type=str, default='logs', help='Directory to save results')
parser.add_argument('--model', type=str, default='RandomForest', choices=['LogisticRegression', 'RandomForest'], 
                    help='Model to use for classification (default: RandomForest)')
parser.add_argument('--val_test_split_ratio', type=float, default=0, 
                    help='Validation/Test split ratio (default: 0)')
parser.add_argument('--generate_plots', action='store_true', help='Generate plots for the final experiment')
parser.add_argument('--njobs', type=int, default=12, help='Number of parallel jobs')
parser.add_argument('--chunk_size', type=int, default=1000, help='Chunk size for processing')
parser.add_argument('--bands_file_path', type=str, 
                   default="data/bands_downsample_100.npy",
                   help='Path to S2 bands file')
parser.add_argument('--label_file_path', type=str,
                   default="data/fieldtype_17classes_downsample_100.npy",
                   help='Path to labels file')
parser.add_argument('--sar_asc_bands_file_path', type=str,
                   default="data/sar_ascending_downsample_100.npy",
                   help='Path to SAR ascending bands file')
parser.add_argument('--sar_desc_bands_file_path', type=str,
                   default="data/sar_descending_downsample_100.npy",
                   help='Path to SAR descending bands file')
args = parser.parse_args()

# Create results directory if it doesn't exist
os.makedirs(args.result_dir, exist_ok=True)

# Generate timestamp for unique filenames
timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
log_file = os.path.join(args.result_dir, f"feature_analysis_{args.model}_{timestamp}.log")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
)
logging.info("Program started.")

# Configuration parameters from command line arguments
PIXELS_PER_CLASS = args.sample_per_pixel
VAL_TEST_SPLIT_RATIO = args.val_test_split_ratio
MODEL = args.model
NUM_EXPERIMENT = args.num_experiment
njobs = args.njobs
chunk_size = args.chunk_size
bands_file_path = args.bands_file_path
label_file_path = args.label_file_path
sar_asc_bands_file_path = args.sar_asc_bands_file_path
sar_desc_bands_file_path = args.sar_desc_bands_file_path

# CSV file path for results
result_csv_path = os.path.join(args.result_dir, f"{PIXELS_PER_CLASS}_{NUM_EXPERIMENT}_{MODEL}.csv")

# Sentinel-2 normalization parameters
S2_BAND_MEAN = np.array([1711.0938, 1308.8511, 1546.4543, 3010.1293, 3106.5083,
                        2068.3044, 2685.0845, 2931.5889, 2514.6928, 1899.4922], dtype=np.float32)
S2_BAND_STD = np.array([1926.1026, 1862.9751, 1803.1792, 1741.7837, 1677.4543,
                       1888.7862, 1736.3090, 1715.8104, 1514.5199, 1398.4779], dtype=np.float32)
S1_BAND_MEAN = np.array([5484.0407,3003.7812], dtype=np.float32)
S1_BAND_STD = np.array([1871.2334,1726.0670], dtype=np.float32)

# Class names for visualization
class_names = [
    "Legume",
    "Soy",
    "Summer Grain",
    "Winter Grain",
    "Corn",
    "Sunflower",
    "Mustard",
    "Potato",
    "Beet",
    "Squash",
    "Grapes",
    "Tree Fruit",
    "Cover Crop",
    "Grass",
    "Fallow",
    "Other (Plants)",
    "Other (Non Plants)"
]

# -----------------------------------------------------------------------------
# Function to evaluate model and collect metrics
# -----------------------------------------------------------------------------
def evaluate_model(y_true, y_pred, valid_classes):
    """
    Evaluate model performance and return detailed metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        valid_classes: List of valid class IDs
        
    Returns:
        metrics (dict): Dictionary containing performance metrics
    """
    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_f1 = f1_score(y_true, y_pred, average='weighted')
    overall_precision = precision_score(y_true, y_pred, average='weighted')
    overall_recall = recall_score(y_true, y_pred, average='weighted')
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Get detailed classification report
    class_report = classification_report(y_true, y_pred, output_dict=True)
    
    # Create results dictionary
    metrics = {
        'overall_accuracy': overall_accuracy,
        'overall_f1': overall_f1,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'balanced_acc': balanced_acc
    }
    
    # Add per-class metrics
    for cls in sorted(valid_classes):
        if str(cls) in class_report:
            cls_metrics = class_report[str(cls)]
            metrics[f'class_{cls}_precision'] = cls_metrics['precision']
            metrics[f'class_{cls}_recall'] = cls_metrics['recall']
            metrics[f'class_{cls}_f1'] = cls_metrics['f1-score']
    
    return metrics

# -----------------------------------------------------------------------------
# Main function for running experiments
# -----------------------------------------------------------------------------
def run_experiment(exp_id):
    """
    Run a single experiment with random sampling
    
    Args:
        exp_id: Experiment ID number
        
    Returns:
        result_dict: Dictionary containing experiment results
    """
    logging.info(f"\n{'='*80}\nStarting Experiment {exp_id+1}/{NUM_EXPERIMENT}\n{'='*80}")
    
    # ----------------- Data Loading and Preprocessing -----------------
    logging.info(f"Experiment {exp_id+1}: Pixels per class: {PIXELS_PER_CLASS}")
    logging.info(f"Experiment {exp_id+1}: Validation/Test split ratio: {VAL_TEST_SPLIT_RATIO}")
    logging.info(f"Experiment {exp_id+1}: Selected model: {MODEL}")
    
    # Create pixel coordinate indices
    pixel_indices = []
    for y in range(H):
        for x in range(W):
            cls = labels[y, x]
            if cls in valid_classes:
                pixel_indices.append((y, x, cls))
    
    # Group coordinates by class
    class_to_pixels = {cls: [] for cls in valid_classes}
    for y, x, cls in pixel_indices:
        class_to_pixels[cls].append((y, x))
    
    # Create training set mask
    train_mask = np.zeros((H, W), dtype=bool)
    val_mask = np.zeros((H, W), dtype=bool)
    test_mask = np.zeros((H, W), dtype=bool)
    
    # Select training pixels for each class - randomly select different pixels in each experiment
    for cls, pixels in class_to_pixels.items():
        class_name = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
        num_pixels = len(pixels)
        
        if num_pixels < PIXELS_PER_CLASS:
            logging.warning(f"Experiment {exp_id+1}: Only {num_pixels} pixels available for class {cls} ({class_name}), which is less than the requested {PIXELS_PER_CLASS}")
            # If insufficient pixels, use all for training
            selected_indices = list(range(num_pixels))
        else:
            # Randomly select pixels - each experiment will select different pixels
            selected_indices = np.random.choice(num_pixels, PIXELS_PER_CLASS, replace=False)
        
        # Mark selected pixels as training set
        for idx in selected_indices:
            y, x = pixels[idx]
            train_mask[y, x] = True
        
        # Assign remaining pixels to validation and test sets
        remaining_indices = [i for i in range(num_pixels) if i not in selected_indices]
        np.random.shuffle(remaining_indices)
        
        if VAL_TEST_SPLIT_RATIO == 0:
            # Assign all to test set
            val_indices = []
            test_indices = remaining_indices
        elif VAL_TEST_SPLIT_RATIO == 1:
            # Assign all to validation set
            val_indices = remaining_indices
            test_indices = []
        else:
            # Split by ratio
            split_idx = int(len(remaining_indices) * VAL_TEST_SPLIT_RATIO)
            val_indices = remaining_indices[:split_idx]
            test_indices = remaining_indices[split_idx:]
        
        # Validation set
        for idx in val_indices:
            y, x = pixels[idx]
            val_mask[y, x] = True
            
        # Test set
        for idx in test_indices:
            y, x = pixels[idx]
            test_mask[y, x] = True
    
    # Count pixels in each set
    train_pixels = np.sum(train_mask)
    val_pixels = np.sum(val_mask)
    test_pixels = np.sum(test_mask)
    logging.info(f"Experiment {exp_id+1}: Train pixels: {train_pixels}, Val pixels: {val_pixels}, Test pixels: {test_pixels}")
    
    # ----------------- Feature Extraction and Data Processing -----------------
    def process_chunk(h_start, h_end, w_start, w_end, file_path):
        logging.info(f"Experiment {exp_id+1}: Processing chunk: h[{h_start}:{h_end}], w[{w_start}:{w_end}]")
        
        # Get masks for current chunk
        chunk_train_mask = train_mask[h_start:h_end, w_start:w_end]
        chunk_val_mask = val_mask[h_start:h_end, w_start:w_end]
        chunk_test_mask = test_mask[h_start:h_end, w_start:w_end]
        chunk_labels = labels[h_start:h_end, w_start:w_end]
        
        # Read S2 data
        tile_chunk = np.load(file_path)[:, h_start:h_end, w_start:w_end, :] # (time, h, w, bands)
        tile_chunk = (tile_chunk - S2_BAND_MEAN) / S2_BAND_STD
        # Reshape data
        time_steps, h, w, bands = tile_chunk.shape
        s2_band_chunk = tile_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
        
        # Read S1 data
        sar_asc_chunk = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
        sar_desc_chunk = np.load(sar_desc_bands_file_path)[:, h_start:h_end, w_start:w_end]
        # Concatenate along time dimension
        sar_chunk = np.concatenate((sar_asc_chunk, sar_desc_chunk), axis=0)
        # Normalize
        sar_chunk = (sar_chunk - S1_BAND_MEAN) / S1_BAND_STD
        # Reshape data
        time_steps, h, w, bands = sar_chunk.shape
        sar_band_chunk = sar_chunk.transpose(1, 2, 0, 3).reshape(-1, time_steps * bands) # (h*w, time_steps*bands)
        
        # Concatenate S2 and S1 features
        X_chunk = np.concatenate((s2_band_chunk, sar_band_chunk), axis=1) # (h*w, time_steps*bands*2)
        y_chunk = chunk_labels.ravel()
        
        # Convert chunk masks to 1D
        train_mask_1d = chunk_train_mask.ravel()
        val_mask_1d = chunk_val_mask.ravel()
        test_mask_1d = chunk_test_mask.ravel()
        
        # Keep only valid class data
        valid_mask = np.isin(y_chunk, list(valid_classes))
        
        # Extract training, validation and test data
        X_train_chunk = X_chunk[valid_mask & train_mask_1d]
        y_train_chunk = y_chunk[valid_mask & train_mask_1d]
        
        X_val_chunk = X_chunk[valid_mask & val_mask_1d]
        y_val_chunk = y_chunk[valid_mask & val_mask_1d]
        
        X_test_chunk = X_chunk[valid_mask & test_mask_1d]
        y_test_chunk = y_chunk[valid_mask & test_mask_1d]
        
        return (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk, X_test_chunk, y_test_chunk)
    
    # Parallel data processing
    chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
             for h in range(0, H, chunk_size)
             for w in range(0, W, chunk_size)]
    logging.info(f"Experiment {exp_id+1}: Total chunks: {len(chunks)}")
    
    results = Parallel(n_jobs=njobs)(
        delayed(process_chunk)(h_start, h_end, w_start, w_end, bands_file_path)
        for h_start, h_end, w_start, w_end in chunks
    )
    
    # Merge results - handle potentially empty validation or test sets
    X_train_parts = [res[0] for res in results if res[0].size > 0]
    y_train_parts = [res[1] for res in results if res[1].size > 0]
    X_val_parts = [res[2] for res in results if res[2].size > 0]
    y_val_parts = [res[3] for res in results if res[3].size > 0]
    X_test_parts = [res[4] for res in results if res[4].size > 0]
    y_test_parts = [res[5] for res in results if res[5].size > 0]
    
    # Safely merge results, handling empty lists
    feature_dim = None
    if X_train_parts:
        X_train = np.vstack(X_train_parts)
        y_train = np.hstack(y_train_parts)
        feature_dim = X_train.shape[1]
    else:
        X_train = np.array([]).reshape(0, 0)
        y_train = np.array([])
    
    if X_val_parts:
        X_val = np.vstack(X_val_parts)
        y_val = np.hstack(y_val_parts)
        if feature_dim is None:
            feature_dim = X_val.shape[1]
    else:
        # If validation set is empty, create correctly shaped but empty array
        X_val = np.array([]).reshape(0, feature_dim if feature_dim is not None else 0)
        y_val = np.array([])
    
    if X_test_parts:
        X_test = np.vstack(X_test_parts)
        y_test = np.hstack(y_test_parts)
        if feature_dim is None:
            feature_dim = X_test.shape[1]
    else:
        # If test set is empty, create correctly shaped but empty array
        X_test = np.array([]).reshape(0, feature_dim if feature_dim is not None else 0)
        y_test = np.array([])
    
    logging.info(f"Experiment {exp_id+1}: Data split summary:")
    logging.info(f"Experiment {exp_id+1}: Train set: {X_train.shape[0]} samples")
    logging.info(f"Experiment {exp_id+1}: Validation set: {X_val.shape[0]} samples")
    logging.info(f"Experiment {exp_id+1}: Test set: {X_test.shape[0]} samples")
    
    # Print data shapes
    if X_train.size > 0:
        logging.info(f"Experiment {exp_id+1}: X_train shape: {X_train.shape}")
    
    # ----------------- Model Training -----------------
    logging.info(f"\nExperiment {exp_id+1}: Training {MODEL}...")
    
    if MODEL == "LogisticRegression":
        model = LogisticRegression(
            multi_class='multinomial',
            solver='lbfgs',
            C=1e4,
            max_iter=100000,
            n_jobs=njobs,
            random_state=None  # Remove fixed seed for bootstrap sampling
        )
    elif MODEL == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=njobs,
            random_state=None  # Remove fixed seed for bootstrap sampling
        )
    else:
        raise ValueError(f"Unknown model type: {MODEL}. Use 'LogisticRegression' or 'RandomForest'")
    
    model.fit(X_train, y_train)
    
    # ----------------- Evaluation -----------------
    # Initialize result dictionary
    result_row = {
        'experiment_id': exp_id + 1,
        'sample_per_pixel': PIXELS_PER_CLASS,
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'val_size': X_val.shape[0]
    }
    
    # Evaluate only when test set is non-empty
    if X_test.shape[0] > 0:
        logging.info(f"Experiment {exp_id+1}: Evaluating model on test set...")
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred, valid_classes)
        result_row.update(metrics)
        
        logging.info(f"Experiment {exp_id+1}: Test Accuracy: {metrics['overall_accuracy']:.4f}, "
                    f"Test F1: {metrics['overall_f1']:.4f}, "
                    f"Balanced Acc: {metrics['balanced_acc']:.4f}")
    elif X_val.shape[0] > 0:
        # If test set is empty but validation set is non-empty, evaluate on validation set
        logging.info(f"Experiment {exp_id+1}: Test set is empty. Evaluating model on validation set...")
        y_pred = model.predict(X_val)
        metrics = evaluate_model(y_val, y_pred, valid_classes)
        result_row.update(metrics)
        
        logging.info(f"Experiment {exp_id+1}: Validation Accuracy: {metrics['overall_accuracy']:.4f}, "
                    f"Validation F1: {metrics['overall_f1']:.4f}, "
                    f"Balanced Acc: {metrics['balanced_acc']:.4f}")
    else:
        logging.info(f"Experiment {exp_id+1}: No test or validation set available for evaluation.")
        
    # Only generate maps for the last experiment if requested
    if args.generate_plots and exp_id == NUM_EXPERIMENT - 1:
        generate_maps(model, train_mask, val_mask, test_mask, exp_id)
    
    return result_row

# -----------------------------------------------------------------------------
# Function to generate maps and plots (only for the last experiment if requested)
# -----------------------------------------------------------------------------
def generate_maps(model, train_mask, val_mask, test_mask, exp_id):
    """Generate prediction maps and evaluation plots."""
    logging.info(f"Experiment {exp_id+1}: Generating visualization maps...")
    
    # Generate prediction map
    pred_map = np.zeros_like(labels)
    
    # Create training/testing split map for visualization
    train_test_mask = np.zeros((H, W), dtype=np.int8)
    train_test_mask[train_mask] = 1  # Training set
    train_test_mask[val_mask] = 2    # Validation set
    train_test_mask[test_mask] = 3   # Test set
    
    # Define color maps and visualization functions
    def get_color_palette(n_classes):
        """Generate a color palette for classification maps."""
        # Start with tab20 which has 20 distinct colors
        base_cmap = plt.cm.get_cmap('tab20')
        colors = [base_cmap(i) for i in range(20)]
        
        # If we need more colors, add from other colormaps
        if n_classes > 20:
            extra_cmap = plt.cm.get_cmap('tab20b')
            colors.extend([extra_cmap(i) for i in range(n_classes - 20)])
        
        # Return only the number of colors we need
        return colors[:n_classes]
    
    # Setup for visualization
    # Add 1 to max class for background (0)
    max_class = max(valid_classes) 
    n_classes = len(valid_classes)
    
    # Generate color palette
    colors = get_color_palette(max_class + 1)  # +1 for background class (0)
    # Set background (0) to white
    colors[0] = (1, 1, 1, 1)  # White for background
    
    # Create colormap
    cmap = ListedColormap(colors)
    
    def plot_classification_map(data, title, cmap, class_names, save_path, figsize=(12, 10)):
        """Create a nicely formatted classification map without colorbar."""
        plt.figure(figsize=figsize, dpi=300)
        
        # Set up the plot with publication quality
        plt.rcParams.update({
            'font.family': 'sans-serif',  # Use a generic font family available everywhere
            'font.size': 12,
            'axes.linewidth': 1.5
        })
        
        # Plot the data
        im = plt.imshow(data, cmap=cmap, interpolation='nearest')
        
        # Create a legend with class names
        if class_names:
            # Get the number of unique classes in the data
            unique_classes = sorted(np.unique(data))
            # Filter out 0 if it's background
            if 0 in unique_classes and len(unique_classes) > 1:
                unique_classes = [c for c in unique_classes if c > 0]
            
            # Create legend patches for each class
            legend_patches = []
            for cls in unique_classes:
                if cls == 0:
                    continue  # Skip background
                if cls <= len(class_names):
                    # Use class color from colormap
                    color = cmap(cls / max(unique_classes))
                    label = class_names[cls-1] if cls-1 < len(class_names) else f"Class {cls}"
                    legend_patches.append(mpatches.Patch(color=color, label=label))
            
            # Add legend outside the plot with larger text and make it more prominent
            plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), 
                    loc='upper left', fontsize=14, frameon=True, fancybox=True, 
                    shadow=True, title="Classes", title_fontsize=15)
        
        # Add title and style adjustments
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved classification map to {save_path}")
    
    # 1. Original Ground Truth Map
    plot_classification_map(
        labels, 
        f"Ground Truth Land Cover Classification", 
        cmap, 
        class_names, 
        os.path.join(args.result_dir, f"ground_truth_map_exp{exp_id+1}.png")
    )
    
    # 2. Training/Testing Split Map
    train_test_cmap = ListedColormap(['white', 'blue', 'green', 'red'])  # White for background, blue for training, green for validation, red for testing
    train_test_names = ["Background", "Training Set", "Validation Set", "Testing Set"]
    
    plot_classification_map(
        train_test_mask, 
        f"Training, Validation and Testing Sample Distribution", 
        train_test_cmap, 
        train_test_names, 
        os.path.join(args.result_dir, f"train_test_split_map_exp{exp_id+1}.png")
    )
    
    # ----------------- Generating Prediction Map -----------------
    # Optimized batch prediction for a whole chunk
    def batch_predict_chunk(h_start, h_end, w_start, w_end):
        """Process and predict a chunk of the image more efficiently."""
        # Create mask for valid classes in this chunk
        chunk_labels = labels[h_start:h_end, w_start:w_end]
        chunk_train_mask = train_mask[h_start:h_end, w_start:w_end]
        
        # Create empty prediction array for this chunk
        chunk_pred = np.zeros_like(chunk_labels)
        
        # Identify valid pixels that need prediction
        valid_mask = np.isin(chunk_labels, list(valid_classes))
        
        # Training pixels directly copy labels
        train_pixels_mask = valid_mask & chunk_train_mask
        chunk_pred[train_pixels_mask] = chunk_labels[train_pixels_mask]
        
        # Pixels that need prediction (non-training valid pixels)
        predict_mask = valid_mask & ~chunk_train_mask
        
        # If there are no pixels to predict, return early
        if not np.any(predict_mask):
            return h_start, h_end, w_start, w_end, chunk_pred
        
        # Get coordinates of pixels that need prediction
        h_indices, w_indices = np.where(predict_mask)
        
        # Load data for feature extraction (only once per chunk)
        s2_data = np.load(bands_file_path)[:, h_start:h_end, w_start:w_end, :]
        sar_asc_data = np.load(sar_asc_bands_file_path)[:, h_start:h_end, w_start:w_end]
        sar_desc_data = np.load(sar_desc_bands_file_path)[:, h_start:h_end, w_start:w_end]
        
        # Batch size for processing within chunk
        batch_size = 1000
        for i in range(0, len(h_indices), batch_size):
            batch_h = h_indices[i:i+batch_size]
            batch_w = w_indices[i:i+batch_size]
            
            # Extract features for this batch of pixels
            batch_features = []
            for j in range(len(batch_h)):
                h_idx, w_idx = batch_h[j], batch_w[j]
                
                # S2 feature extraction
                s2_pixel = s2_data[:, h_idx, w_idx, :]
                s2_norm = (s2_pixel - S2_BAND_MEAN) / S2_BAND_STD
                s2_features = s2_norm.reshape(-1)
                
                # S1 feature extraction
                sar_asc_pixel = sar_asc_data[:, h_idx, w_idx]
                sar_desc_pixel = sar_desc_data[:, h_idx, w_idx]
                sar_pixel = np.concatenate((sar_asc_pixel, sar_desc_pixel))
                sar_norm = (sar_pixel - S1_BAND_MEAN) / S1_BAND_STD
                sar_features = sar_norm.reshape(-1)
                
                # Combine features
                features = np.concatenate((s2_features, sar_features))
                batch_features.append(features)
            
            # Convert to numpy array
            batch_features = np.array(batch_features)
            
            # Batch prediction
            batch_preds = model.predict(batch_features)
            
            # Place predictions into chunk
            for j in range(len(batch_h)):
                h_idx, w_idx = batch_h[j], batch_w[j]
                chunk_pred[h_idx, w_idx] = batch_preds[j]
        
        return h_start, h_end, w_start, w_end, chunk_pred
    
    # Define chunks for parallel processing of prediction map
    pred_chunks = [(h, min(h+chunk_size, H), w, min(w+chunk_size, W))
                for h in range(0, H, chunk_size)
                for w in range(0, W, chunk_size)]
    
    # Process prediction map in parallel
    logging.info(f"Experiment {exp_id+1}: Processing prediction map in parallel...")
    start_time = time.time()
    
    pred_results = Parallel(n_jobs=njobs)(
        delayed(batch_predict_chunk)(h_start, h_end, w_start, w_end)
        for h_start, h_end, w_start, w_end in pred_chunks
    )
    
    # Combine prediction results
    for h_start, h_end, w_start, w_end, chunk_pred in pred_results:
        pred_map[h_start:h_end, w_start:w_end] = chunk_pred
    
    end_time = time.time()
    logging.info(f"Experiment {exp_id+1}: Prediction map generation completed in {end_time - start_time:.2f} seconds")
    
    # 3. Model Prediction Map
    model_name = MODEL
    plot_classification_map(
        pred_map, 
        f"{model_name} Classification Predictions", 
        cmap, 
        class_names, 
        os.path.join(args.result_dir, f"prediction_map_{model_name.lower()}_exp{exp_id+1}.png")
    )
    
    # Generate a composite map that shows the differences between prediction and ground truth
    diff_map = np.zeros_like(labels)
    
    # Calculate difference map
    for h in range(H):
        for w in range(W):
            if labels[h, w] in valid_classes and pred_map[h, w] > 0:
                if train_mask[h, w]:
                    # Training pixels - should match ground truth (but mark differently)
                    diff_map[h, w] = 1  # Training pixel
                else:
                    # Test/Val pixels - check if prediction matches ground truth
                    diff_map[h, w] = 2 if pred_map[h, w] == labels[h, w] else 3  # 2=correct, 3=incorrect
    
    diff_cmap = ListedColormap(['white', 'blue', 'lightgray', 'red'])  # White=background, blue=training, gray=correct, red=incorrect
    diff_names = ["Background", "Training", "Correct Prediction", "Incorrect Prediction"]
    
    plot_classification_map(
        diff_map, 
        f"{model_name} Prediction Accuracy", 
        diff_cmap, 
        diff_names, 
        os.path.join(args.result_dir, f"prediction_difference_map_{model_name.lower()}_exp{exp_id+1}.png")
    )

# -----------------------------------------------------------------------------
# Main function: Load data, run multiple experiments and save results
# -----------------------------------------------------------------------------
def main():
    global labels, H, W, valid_classes
    
    # ----------------- Data Loading -----------------
    logging.info(f"Loading labels...")
    labels = np.load(label_file_path).astype(np.int64)
    H, W = labels.shape
    logging.info(f"Data dimensions: {H}x{W}")
    
    # Valid class selection
    logging.info("Identifying valid classes...")
    class_counts = Counter(labels.ravel())
    valid_classes = {cls for cls, count in class_counts.items() if count >= 2}
    valid_classes.discard(0)  # Remove background class
    logging.info(f"Valid classes: {sorted(valid_classes)}")
    
    # Create list to store all experiment results
    all_results = []
    
    # Run multiple experiments
    for exp_id in range(NUM_EXPERIMENT):
        result_row = run_experiment(exp_id)
        all_results.append(result_row)
    
    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(result_csv_path, index=False)
    logging.info(f"All experiment results saved to {result_csv_path}")
    
    # Print statistics of results
    logging.info("\nExperiment Statistics:")
    for metric in ['overall_accuracy', 'overall_f1', 'overall_precision', 'overall_recall', 'balanced_acc']:
        if metric in results_df.columns:
            values = results_df[metric].values
            logging.info(f"{metric}: mean={np.mean(values):.4f}, std={np.std(values):.4f}, "
                        f"min={np.min(values):.4f}, max={np.max(values):.4f}")

if __name__ == "__main__":
    main()
    print(f"Process finished. Logs saved to: {log_file}")
    print(f"All experiment results saved to: {result_csv_path}")
    if args.generate_plots:
        print("Classification maps saved in the results directory.")