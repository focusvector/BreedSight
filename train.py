# =========================================
# 0. Imports
# =========================================
import json # Imports the JSON module for working with JSON files (e.g., saving the class map).
import os # Imports the OS module for interacting with the operating system (not directly used here but good practice).
import pathlib # Imports the pathlib module for an object-oriented way to handle filesystem paths.
import random # Imports the random module for generating random numbers (used for augmentations).
import time # Imports the time module for timing operations like epoch duration.
import zipfile # Imports the zipfile module for working with ZIP archives.

import numpy as np # Imports NumPy for numerical operations, especially for data augmentation calculations.
import torch # Imports the main PyTorch library.
import torch.nn as nn # Imports PyTorch's neural network module for layers and loss functions.
import torch.optim as optim # Imports PyTorch's optimization module for optimizers like Adam.
from torch.utils.data import DataLoader, random_split # Imports DataLoader for batching and random_split for creating validation sets.
from torchvision import transforms # Imports torchvision's transforms module for common image transformations.

from config import Config # Imports the Config class from our local config.py file.
from dataset_utils import FusedDataset, prepare_fused_samples # Imports our custom dataset and preparation function.
from engine import train_one_epoch, validate # Imports our core training and validation functions from engine.py.
from model import build_model # Imports our model-building function from model.py.
from plotting_utils import LivePlot # Imports our live plotting class from plotting_utils.py.

# =========================================
# 1. Helper Functions
# =========================================
def set_seed(seed):
    """Sets random seeds for all relevant libraries to ensure reproducibility."""
    random.seed(seed) # Sets the seed for Python's built-in random module.
    np.random.seed(seed) # Sets the seed for NumPy's random number generator.
    torch.manual_seed(seed) # Sets the seed for PyTorch's CPU operations.
    if torch.cuda.is_available(): # Checks if a CUDA-enabled GPU is available.
        torch.cuda.manual_seed_all(seed) # Sets the seed for all available GPUs.
        torch.backends.cudnn.benchmark = True # Enables cuDNN's auto-tuner, which can speed up training if input sizes don't change.

def auto_detect_and_prepare_datasets(root_dir):
    """Scans a root directory for datasets, unzipping any .zip files if necessary."""
    print(f"🔍 Scanning for datasets in: {root_dir}") # Prints a status message indicating the scan location.
    datasets_root_path = pathlib.Path(root_dir) # Creates a Path object for easier manipulation.
    valid_dataset_paths = [] # Initializes a list to store the paths of valid dataset folders.
    if not datasets_root_path.is_dir(): # Checks if the provided root path is not a directory.
        print(f"❌ Error: Root datasets directory not found at '{root_dir}'") # Prints an error message if the path is invalid.
        return [] # Returns an empty list to indicate failure.
    for item_path in datasets_root_path.iterdir(): # Iterates over every file and folder in the root directory.
        if item_path.is_file() and item_path.suffix == '.zip': # Checks if the item is a file with a .zip extension.
            extracted_folder_path = datasets_root_path / item_path.stem # Defines the path where the ZIP file will be extracted.
            if not extracted_folder_path.is_dir(): # Checks if the destination folder does not already exist.
                print(f"  - Found ZIP file: '{item_path.name}'. Extracting...") # Announces the extraction process.
                with zipfile.ZipFile(item_path, 'r') as zip_ref: # Opens the ZIP file in read mode.
                    zip_ref.extractall(extracted_folder_path) # Extracts all contents of the ZIP file to the destination folder.
                print(f"  ✔ Extracted to: '{extracted_folder_path.name}'") # Confirms successful extraction.
            else: # If the destination folder already exists.
                print(f"  - Found ZIP file: '{item_path.name}'. Matching folder already exists.") # Informs the user that extraction is skipped.
            valid_dataset_paths.append(str(extracted_folder_path)) # Adds the path of the extracted folder to the list of valid datasets.
        elif item_path.is_dir(): # Checks if the item is a directory.
            print(f"  - Found dataset folder: '{item_path.name}'") # Announces that a dataset folder was found.
            valid_dataset_paths.append(str(item_path)) # Adds the directory path to the list of valid datasets.
    print(f"\n✅ Finished detection. Using {len(valid_dataset_paths)} dataset(s): {valid_dataset_paths}\n") # Prints a summary of all found datasets.
    return valid_dataset_paths # Returns the list of valid dataset paths.

# =========================================
# 2. Data Preparation Function
# =========================================
def prepare_data(cfg):
    """Fuses datasets, defines transforms, and creates DataLoaders using a config object."""
    dataset_directories = auto_detect_and_prepare_datasets(cfg.DATASETS_ROOT_DIR) # First, find all dataset directories.
    # Note: class_weights_list is now expected to be None from this function call.
    train_samples, valid_samples, unified_class_map, _ = prepare_fused_samples( # Call the main data fusion function.
        dataset_dirs=dataset_directories, 
        selection_mode=cfg.CLASS_SELECTION_MODE,
        top_n=cfg.TOP_N_CLASSES,
        specific_classes=cfg.SPECIFIC_CLASSES
    )
    if not train_samples: raise ValueError("Dataset preparation resulted in no training samples.") # If no samples are found, raise an error to stop the script.
    num_classes = len(unified_class_map) # Get the total number of classes being used.
    with open(cfg.CLASS_MAP_PATH, "w") as f: json.dump(unified_class_map, f, indent=4) # Save the class-to-integer mapping to a JSON file.
    print(f"Saved class mapping for {num_classes} classes to {cfg.CLASS_MAP_PATH}") # Confirm that the class map was saved.


    train_transform = transforms.Compose([ # Define the sequence of transformations for the training data.
        transforms.RandomResizedCrop(cfg.IMAGE_SIZE, scale=(0.8, 1.0)), # Randomly crop and resize the image.
        transforms.RandomHorizontalFlip(), # Randomly flip the image horizontally.
        transforms.RandomRotation(15), # Randomly rotate the image by up to 15 degrees.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Randomly change the brightness, contrast, and saturation.
        transforms.ToTensor(), # Convert the PIL Image to a PyTorch tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Normalize the tensor with ImageNet mean and std.
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.25)), # Randomly erase a rectangular region in the image.
    ])
    val_transform = transforms.Compose([ # Define the sequence of transformations for the validation data (no augmentation).
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)), # Resize the image to the required input size.
        transforms.ToTensor(), # Convert the PIL Image to a PyTorch tensor.
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), # Normalize the tensor with ImageNet mean and std.
    ])
    
    if not valid_samples: # If no pre-split validation set was found.
        print("No pre-split validation set found. Performing 80/20 random split on the list of training samples.") # Announce the split.
        train_size = int(0.8 * len(train_samples)) # Calculate the size of the training set (80%).
        val_size = len(train_samples) - train_size # The remaining 20% is for validation.
        generator = torch.Generator().manual_seed(cfg.RANDOM_SEED) # Create a generator for a reproducible random split.
        train_indices, val_indices = random_split(range(len(train_samples)), [train_size, val_size], generator=generator) # Perform the split on the indices.
        final_train_samples = [train_samples[i] for i in train_indices] # Create the final training list from the split indices.
        final_valid_samples = [train_samples[i] for i in val_indices] # Create the final validation list from the split indices.
    else: # If a pre-split validation set was found.
        print("Using pre-split validation set found in dataset folders.") # Announce that the pre-split set is being used.
        final_train_samples = train_samples # Use the training samples as is.
        final_valid_samples = valid_samples # Use the validation samples as is.

    # --- FIX: Calculate class weights based on the FINAL training set ---
    print("\n⚖️ Calculating class weights for the final training set...") # Announce the weight calculation.
    from collections import Counter # Import Counter for this specific task.
    label_counts = Counter([label for _, label in final_train_samples]) # Count the occurrences of each class label in the final training set.
    total_samples = len(final_train_samples) # Get the total number of training samples.
    class_weights = torch.zeros(num_classes) # Initialize a tensor of zeros to hold the weights.
    for i in range(num_classes): # Loop through each possible class index.
        count = label_counts.get(i, 0) # Get the count for the current class, defaulting to 0 if not present.
        if count == 0: # If a class has no samples in the training set.
            class_weights[i] = 1.0 # Assign a neutral weight of 1.0.
        else: # If the class has samples.
            class_weights[i] = total_samples / (num_classes * count) # Calculate the inverse frequency weight.
    class_weights = class_weights.to(cfg.DEVICE) # Move the weights tensor to the target device (CPU or GPU).
    print(f"Calculated weights: {class_weights}") # Print the final calculated weights.

    train_dataset = FusedDataset(final_train_samples, transform=train_transform, preload_into_ram=cfg.PRELOAD_DATASET_INTO_RAM, safety_margin=cfg.MEMORY_SAFETY_MARGIN) # Create the training dataset object.
    val_dataset = FusedDataset(final_valid_samples, transform=val_transform, preload_into_ram=cfg.PRELOAD_DATASET_INTO_RAM, safety_margin=cfg.MEMORY_SAFETY_MARGIN) # Create the validation dataset object.

    # --- FIX: Optimize num_workers based on pre-loading status ---
    num_workers = 4 if not train_dataset.should_preload else 0 # Set num_workers to 0 if data is in RAM, otherwise use 4.
    if num_workers == 0: # If num_workers was set to 0.
        print("Dataset is pre-loaded into RAM. Setting num_workers to 0 for optimal performance.") # Inform the user about the optimization.

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True) # Create the DataLoader for the training set.
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True) # Create the DataLoader for the validation set.
    
    return train_loader, val_loader, num_classes, class_weights # Return all the necessary objects for the main training loop.

# =========================================
# 3. Main Execution Block
# =========================================
if __name__ == "__main__": # This block ensures the code runs only when the script is executed directly.
    cfg = Config() # Create an instance of our configuration class.
    set_seed(cfg.RANDOM_SEED) # Set the random seed for reproducibility.
    
    if torch.cuda.is_available(): # Check if a CUDA-enabled GPU is available.
        print(f"✅ GPU found: {torch.cuda.get_device_name(0)}") # If yes, print the name of the GPU.
    else: # If not.
        print("❌ No GPU found. Training will run on CPU.") # Inform the user that the CPU will be used.
    print(f"Using device: {cfg.DEVICE}") # Print the selected device.

    train_loader, val_loader, num_classes, class_weights = prepare_data(cfg) # Call the data preparation function to get DataLoaders and other info.
    
    model = build_model(num_classes, cfg.DEVICE) # Build the model architecture and move it to the device.
    criterion = nn.CrossEntropyLoss(weight=class_weights) # Define the loss function, passing the calculated class weights to handle imbalance.
    params_to_update = [p for p in model.parameters() if p.requires_grad] # Create a list of only the model parameters that are trainable (not frozen).
    optimizer = optim.Adam(params_to_update, lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY) # Define the Adam optimizer to update the trainable parameters.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3) # Define a scheduler to reduce the learning rate if validation loss plateaus.
    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available()) # Initialize a gradient scaler for automatic mixed precision training.
    
    history = {"train_loss": [], "val_loss": [], "val_acc": []} # Initialize a dictionary to store the training history for plotting.
    live_plot = LivePlot(save_path=cfg.PLOT_SAVE_PATH) # Initialize the live plotting object.
    best_val_loss = float("inf") # Initialize the best validation loss to infinity.
    epochs_no_improve = 0 # Initialize a counter for early stopping.
    
    for epoch in range(cfg.EPOCHS): # Start the main training loop for the specified number of epochs.
        start_time = time.time() # Record the start time of the epoch.
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, cfg.DEVICE) # Perform one epoch of training.
        val_loss, val_acc = validate(model, val_loader, criterion, cfg.DEVICE) # Perform one epoch of validation.
        scheduler.step(val_loss) # Update the learning rate scheduler based on the validation loss.

        history["train_loss"].append(train_loss) # Append the current training loss to the history.
        history["val_loss"].append(val_loss) # Append the current validation loss to the history.
        history["val_acc"].append(val_acc.item()) # Append the current validation accuracy to the history.
        live_plot.update(history) # Update the live plot with the new history data.
        
        elapsed_time = time.time() - start_time # Calculate the time taken for the epoch.
        print(f"Epoch {epoch+1}/{cfg.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Time: {elapsed_time:.1f}s") # Print a summary of the epoch's results.
        
        if val_loss < best_val_loss: # Check if the current validation loss is better than the best one seen so far.
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH) # If yes, save the model's weights.
            best_val_loss = val_loss # Update the best validation loss.
            epochs_no_improve = 0 # Reset the early stopping counter.
            print(f"Model saved. Validation loss improved to {best_val_loss:.4f}") # Print a confirmation message.
        else: # If the validation loss did not improve.
            epochs_no_improve += 1 # Increment the early stopping counter.
            if epochs_no_improve >= cfg.PATIENCE: # Check if the patience limit has been reached.
                print("Early stopping.") # Announce that training is stopping early.
                break # Exit the training loop.
                
    live_plot.save_and_close() # After the loop finishes, save the final plot and close the window.