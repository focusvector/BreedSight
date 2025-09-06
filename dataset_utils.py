import pathlib
from PIL import Image
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm # Import tqdm for the pre-loading progress bar
import os # Import os for checking file sizes.

# Try to import psutil for memory checking; this is a soft dependency.
try:
    import psutil
    
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# This file contains all logic for discovering, selecting, and fusing datasets.

def prepare_fused_samples(dataset_dirs, selection_mode="TOP_N", top_n=10, specific_classes=None):
    """
    Scans multiple dataset directories, selects classes based on the specified mode,
    and returns fused lists of samples for each split, along with a unified class map and class weights.

    Args:
        dataset_dirs (list): A list of paths to dataset root directories.
        selection_mode (str): Either "TOP_N" or "SPECIFIC".
        top_n (int): The number of top classes to select if mode is "TOP_N".
        specific_classes (list): A list of class names to use if mode is "SPECIFIC".

    Returns:
        tuple: (train_samples, valid_samples, unified_class_map, class_weights)
    """
    def _normalize_class_name(name):
        """
        Normalizes a class name to facilitate fusing of similar classes.
        e.g., 'Gir_Cow', 'gir-cattle', 'Gir' -> 'gir'
        """
        name = name.lower() # Convert to lowercase.
        name = name.replace('_', ' ').replace('-', ' ') # Replace separators with spaces.
        # List of common, non-descriptive words to remove.
        common_words = ['cow', 'cattle', 'breed', 'images', 'class']
        # Remove common words from the name.
        words = [word for word in name.split() if word not in common_words]
        return " ".join(words).strip() # Join back and remove leading/trailing whitespace.

    print("🚀 Starting automated dataset discovery and fusion...")
    
    # --- 1. Discover ALL classes and their image paths from all datasets ---
    all_class_paths = {}
    train_samples_map = {} # Maps class_name -> list of training image paths
    valid_samples_map = {} # Maps class_name -> list of validation image paths
    
    for dir_path in dataset_dirs:
        path = pathlib.Path(dir_path)
        if not path.is_dir():
            print(f"⚠️ Warning: Directory not found, skipping: {dir_path}")
            continue
        
        print(f"  - Scanning directory: {path.name}")
        # Check for pre-split structure first
        if (path / "train").is_dir():
            print("    Found pre-split train/valid folders.")
            for split in ["train", "valid"]:
                target_map = train_samples_map if split == "train" else valid_samples_map
                for class_dir in (path / split).iterdir():
                    if class_dir.is_dir():
                        class_name = _normalize_class_name(class_dir.name) # Normalize the class name.
                        target_map.setdefault(class_name, [])
                        all_class_paths.setdefault(class_name, []) # Ensure class exists in all_class_paths
                        image_paths = list(class_dir.glob('*.*'))
                        target_map[class_name].extend(image_paths)
                        all_class_paths[class_name].extend(image_paths) # FIX: Add all images to the master list for counting
        else:
            # Handle simple structure (all images go to training pool)
            for class_dir in path.iterdir():
                if class_dir.is_dir():
                    class_name = _normalize_class_name(class_dir.name) # Normalize the class name.
                    train_samples_map.setdefault(class_name, [])
                    all_class_paths.setdefault(class_name, [])
                    image_paths = list(class_dir.glob('*.*'))
                    train_samples_map[class_name].extend(image_paths)
                    all_class_paths[class_name].extend(image_paths)

    # --- 2. Select which classes to use based on the chosen mode ---
    selected_classes = []
    if selection_mode == "SPECIFIC":
        print(f"\n🎯 Selecting SPECIFIC classes as requested...")
        if not specific_classes: raise ValueError("Selection mode is 'SPECIFIC' but list is empty.")
        # Normalize the user-provided list to match the normalized discovered classes.
        normalized_specific_classes = {_normalize_class_name(c) for c in specific_classes}
        selected_classes = [c for c in all_class_paths.keys() if c in normalized_specific_classes]
    elif selection_mode == "TOP_N":
        print(f"\n🎯 Selecting TOP {top_n} classes by sample count...")
        class_counts = {name: len(paths) for name, paths in all_class_paths.items()}
        sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)
        selected_classes = [name for name, count in sorted_classes[:top_n]]
    else:
        raise ValueError(f"Unknown selection_mode: '{selection_mode}'.")
    
    if not selected_classes:
        print("❌ Error: No classes were selected.")
        return [], [], {}, None
        
    unified_class_map = {name: i for i, name in enumerate(sorted(selected_classes))}
    print("-" * 30)
    print(f"✅ Using {len(unified_class_map)} classes: {list(unified_class_map.keys())}")
    print("-" * 30)

    # --- 3. Create the final sample lists for the SELECTED classes ---
    final_train_samples = []
    final_valid_samples = []

    for class_name, paths in train_samples_map.items():
        if class_name in unified_class_map:
            for path in paths:
                final_train_samples.append((str(path), unified_class_map[class_name]))

    for class_name, paths in valid_samples_map.items():
        if class_name in unified_class_map:
            for path in paths:
                final_valid_samples.append((str(path), unified_class_map[class_name]))

    # --- 4. Calculate Class Weights for the TRAINING data ---
    print("\n⚖️ Calculating class weights for imbalanced dataset...")
    label_counts = Counter([label for _, label in final_train_samples])
    total_samples = len(final_train_samples)
    weights = []
    for i in range(len(unified_class_map)):
        count = label_counts.get(i, 0)
        if count == 0:
            # Assign a neutral weight if a class has no training samples (edge case)
            weights.append(1.0)
        else:
            class_weight = total_samples / (len(unified_class_map) * count)
            weights.append(class_weight)
    
    print(f"✅ Fused {len(final_train_samples)} training images and {len(final_valid_samples)} validation images.")
    return final_train_samples, final_valid_samples, unified_class_map, weights


class FusedDataset(Dataset):
    """
    A custom dataset that works with a list of (image_path, label) tuples.
    Includes an option to preload all images into RAM to speed up training.
    """
    def __init__(self, samples, transform=None, preload_into_ram=False, safety_margin=0.75):
        """
        Initializes the dataset.

        Args:
            samples (list): A list of (image_path, label) tuples.
            transform (callable, optional): A function/transform to apply to the images.
            preload_into_ram (bool): If True, attempts to load all images into memory.
            safety_margin (float): The fraction of available RAM to consider safe for pre-loading.
        """
        def _is_safe_to_preload(samples_to_check, margin):
            """Checks if there is enough available RAM to safely preload the dataset."""
            if not PSUTIL_AVAILABLE:
                print("\n⚠️ Warning: `psutil` not found. Cannot perform memory safety check. Disabling pre-loading.")
                print("   Install it with: pip install psutil")
                return False
            try:
                # Estimate dataset size by summing up the file sizes on disk.
                total_size_bytes = sum(os.path.getsize(path) for path, _ in samples_to_check)
                available_ram_bytes = psutil.virtual_memory().available
                
                # Check if the estimated size is within the safe limit of available RAM.
                if total_size_bytes < available_ram_bytes * margin:
                    print(f"✅ Memory check passed. Dataset size (~{total_size_bytes / 1e9:.2f} GB) is within safe limits.")
                    return True
                else:
                    print("\n❌ Memory safety check failed! Pre-loading aborted.")
                    print(f"   - Estimated dataset size: {total_size_bytes / 1e9:.2f} GB")
                    print(f"   - Available RAM: {available_ram_bytes / 1e9:.2f} GB")
                    print("   - To prevent system instability, training will proceed by loading images from disk.\n")
                    return False
            except Exception as e:
                print(f"⚠️ Warning: Could not perform memory safety check. Error: {e}. Disabling pre-loading.")
                return False

        self.transform = transform
        
        # Perform safety check before attempting to preload.
        self.should_preload = preload_into_ram and _is_safe_to_preload(samples, safety_margin)
        
        # If preloading is enabled and safe, load all images from disk into a list.
        if self.should_preload:
            print(f"🚀 Pre-loading {len(samples)} images into RAM. This may take a moment...")
            # self.samples will store (PIL.Image, label) tuples instead of (path, label).
            self.samples = []
            for path, label in tqdm(samples, desc="Pre-loading"):
                try:
                    # Open the image from the path and convert it to RGB.
                    image = Image.open(path).convert("RGB")
                    self.samples.append((image, label))
                except Exception as e:
                    # If an image is corrupted, it will be skipped during pre-loading.
                    print(f"⚠️ Warning: Could not load image {path}. Error: {e}. Skipping.")
        else:
            # If not preloading, just store the list of file paths.
            self.samples = samples

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a sample from the dataset at the given index.
        """
        if self.should_preload:
            # If preloaded, the image is already a PIL object in memory.
            image, label = self.samples[idx]
        else:
            # If not preloaded, open the image from disk.
            image_path, label = self.samples[idx]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Warning: Could not load image {image_path}. Error: {e}. Skipping.")
                return self.__getitem__((idx + 1) % len(self))
        
        # Apply the transformations if they are defined.
        if self.transform:
            image = self.transform(image)
            
        return image, label