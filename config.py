import torch

class Config:
    """A single class to hold all configuration parameters for the project."""
    # --- Dataset and Class Selection ---
    # A list of paths to the individual dataset folders.
    DATASET_DIRECTORIES = [
        r"D:\dev\sih\datasets\img 13k",
        r"D:\dev\sih\datasets\img 6k"
    ]
    CLASS_SELECTION_MODE = "TOP_N" 
    TOP_N_CLASSES = 20
    SPECIFIC_CLASSES = ["Gir", "Sahiwal", "Jersey"]
    
    # --- Training Hyperparameters ---
    BATCH_SIZE = 32
    IMAGE_SIZE = 299  # InceptionV3 requires 299x299
    EPOCHS = 50
    PATIENCE = 7
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # --- File Paths ---
    MODEL_SAVE_PATH = "final_inception_model.pth"
    CLASS_MAP_PATH = "final_class_mapping.json"
    PLOT_SAVE_PATH = "training_history.png"

    # --- Hardware and Reproducibility ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Automatically select GPU if available, otherwise fall back to CPU.
    RANDOM_SEED = 42 # A fixed seed for all random number generators to ensure results are reproducible.

    # --- Performance Tuning ---
    # LOADING_MODE: "PRELOAD_ALL", "ON_DEMAND", "CHUNKED"
    # - PRELOAD_ALL: Loads entire dataset into RAM. Fastest, but uses most memory.
    # - ON_DEMAND: Loads images from disk one by one. Slowest, but uses least memory.
    # - CHUNKED: Loads dataset in large chunks. A balance between speed and memory.
    LOADING_MODE = "CHUNKED" 
    CHUNK_SIZE = 8000 # Number of images to load into RAM at a time if using CHUNKED mode.
    MEMORY_SAFETY_MARGIN = 0.75 # Use 75% of available RAM at most for pre-loading.
