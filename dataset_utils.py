import pathlib # Imports the pathlib module for object-oriented filesystem paths, making path manipulation easier and more readable.
from PIL import Image # Imports the Image module from the Python Imaging Library (Pillow) for opening and manipulating image files.
from torch.utils.data import Dataset # Imports the base Dataset class from PyTorch, which we will inherit from to create a custom dataset.
from collections import Counter # Imports the Counter class for easily counting hashable objects, used here to count class occurrences.
from tqdm import tqdm # Imports the tqdm library to create smart progress bars for loops, providing visual feedback.
import os # Imports the os module, which provides a way of using operating system dependent functionality like reading file sizes.

# Try to import psutil for memory checking; this is a soft dependency.
try:
	import psutil # Attempt to import the psutil library for system monitoring (e.g., checking available RAM).
	PSUTIL_AVAILABLE = True # If the import succeeds, set a flag to True.
except ImportError: # If psutil is not installed, an ImportError will be raised.
	PSUTIL_AVAILABLE = False # If the import fails, set the flag to False.

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
		# Normalize but ensure non-empty fallback
		n = name.lower().replace('_', ' ').replace('-', ' ') # Convert to lowercase and replace separators with spaces.
		# List of common, non-descriptive words to remove.
		common_words = ['cow', 'cattle', 'breed', 'images', 'class'] # Define a list of common words to filter out.
		# Remove common words from the name.
		words = [w for w in n.split() if w not in common_words] # Create a new list of words, excluding the common ones.
		normalized = " ".join(words).strip() # Join the remaining words back into a string and remove leading/trailing whitespace.
		return normalized if normalized else name.lower().strip() # Return the normalized name, or the original lowercased name if normalization resulted in an empty string.

	print("🚀 Starting automated dataset discovery and fusion...") # Print a status message to the console.

	# --- 1. Discover ALL classes and their image paths from all datasets ---
	all_class_paths = {} # Initialize a dictionary to store all image paths for every class, used for counting.
	train_samples_map = {} # Initialize a dictionary to map class names to their list of training image paths.
	valid_samples_map = {} # Initialize a dictionary to map class names to their list of validation image paths.

	# allow common image extensions
	img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'} # Define a set of valid image file extensions to look for.

	for dir_path in dataset_dirs: # Loop through each dataset directory provided in the input list.
		path = pathlib.Path(dir_path) # Convert the string path to a more robust Path object.
		if not path.is_dir(): # Check if the path is actually a directory.
			print(f"⚠️ Warning: Directory not found, skipping: {dir_path}") # If not, print a warning.
			continue # Skip to the next directory in the loop.

		print(f"  - Scanning directory: {path.name}") # Print the name of the directory being scanned.
		
		# Check for pre-split structure first
		if (path / "train").is_dir():  # Check if a 'train' subdirectory exists, indicating a pre-split dataset.
			print(f"    Found 'train' folder at: {path / 'train'}") # Confirm the pre-split structure.
			for split in ["train", "valid"]: # Loop through the 'train' and 'valid' splits.
				split_path = path / split # Create the full path to the split directory.
				if not split_path.is_dir(): # If a split directory (e.g., 'valid') doesn't exist, skip it.
					continue # Move to the next item in the loop.
					
				print(f"    Processing {split} split...") # Announce which split is being processed.
				# find all class folders directly under split_path
				for class_dir in split_path.iterdir(): # Loop through each item in the split directory.
					if not class_dir.is_dir(): # If the item is not a directory, skip it.
						continue # Move to the next item.
						
					class_name = _normalize_class_name(class_dir.name) # Normalize the directory name to get a standard class name.
					print(f"      Found class: {class_dir.name} -> normalized to: {class_name}") # Print the original and normalized names.
					
					# recursively find all image files
					files = list(class_dir.rglob('*')) # Recursively find all files and folders within the class directory.
					image_files = [f for f in files if f.is_file() and f.suffix.lower() in img_exts] # Filter the list to include only files with valid image extensions.
					
					if not image_files: # If no image files were found in this directory.
						print(f"      ⚠️ No images found in {class_dir}") # Print a warning.
						continue # Skip to the next class directory.
						
					print(f"      ✓ Found {len(image_files)} images") # Print the number of images found.
					
					target_map = train_samples_map if split == "train" else valid_samples_map # Choose the correct dictionary to update based on the split.
					target_map.setdefault(class_name, []) # Ensure the class name exists as a key in the target map, initializing with an empty list if not.
					all_class_paths.setdefault(class_name, []) # Ensure the class name exists in the master path list as well.
					
					image_paths = [str(p) for p in image_files] # Convert the Path objects of the images to strings.
					target_map[class_name].extend(image_paths) # Add the list of image paths to the appropriate split map.
					all_class_paths[class_name].extend(image_paths) # Add the list of image paths to the master list for counting purposes.
		else: # If no 'train' subdirectory was found, treat the dataset as having a simple structure.
			print("    No train/valid split found, treating all subfolders as classes") # Announce the dataset structure type.
			for class_dir in path.iterdir(): # Loop through each item in the main dataset directory.
				if not class_dir.is_dir(): # If the item is not a directory, skip it.
					continue # Move to the next item.
					
				class_name = _normalize_class_name(class_dir.name) # Normalize the directory name to get a standard class name.
				print(f"      Found class: {class_dir.name} -> normalized to: {class_name}") # Print the original and normalized names.
				
				files = list(class_dir.rglob('*')) # Recursively find all files and folders within the class directory.
				image_files = [f for f in files if f.is_file() and f.suffix.lower() in img_exts] # Filter the list to include only files with valid image extensions.
				
				if not image_files: # If no image files were found in this directory.
					print(f"      ⚠️ No images found in {class_dir}") # Print a warning.
					continue # Skip to the next class directory.
					
				print(f"      ✓ Found {len(image_files)} images") # Print the number of images found.
				
				train_samples_map.setdefault(class_name, []) # In a simple structure, all images are considered training samples.
				all_class_paths.setdefault(class_name, []) # Ensure the class name exists in the master path list.
				image_paths = [str(p) for p in image_files] # Convert the Path objects of the images to strings.
				train_samples_map[class_name].extend(image_paths) # Add the image paths to the training map.
				all_class_paths[class_name].extend(image_paths) # Add the image paths to the master list.

	# remove duplicate paths in lists and keep order
	for cmap in (all_class_paths, train_samples_map, valid_samples_map): # Loop through all the dictionaries we've populated.
		for k, lst in list(cmap.items()): # Loop through each class and its list of paths.
			seen = set() # Create an empty set to track seen paths.
			new_list = [] # Create a new empty list for the unique paths.
			for p in lst: # Loop through each path in the current list.
				if p not in seen: # If the path has not been seen before.
					seen.add(p) # Add the path to the set of seen paths.
					new_list.append(p) # Add the path to the new list.
			cmap[k] = new_list # Replace the old list with the new list of unique paths.

	# --- 2. Select which classes to use based on the chosen mode ---
	selected_classes = [] # Initialize an empty list to store the names of the classes that will be used.
	if selection_mode == "SPECIFIC": # If the user chose to select specific classes.
		print(f"\n🎯 Selecting SPECIFIC classes as requested...") # Print a status message.
		if not specific_classes: raise ValueError("Selection mode is 'SPECIFIC' but list is empty.") # If the list is empty, raise an error.
		# Normalize the user-provided list to match the normalized discovered classes.
		normalized_specific_classes = {_normalize_class_name(c) for c in specific_classes} # Create a set of normalized names from the user's list.
		selected_classes = [c for c in all_class_paths.keys() if c in normalized_specific_classes] # Create the final list of selected classes.
	elif selection_mode == "TOP_N": # If the user chose to select the top N most frequent classes.
		print(f"\n🎯 Selecting TOP {top_n} classes by sample count...") # Print a status message.
		class_counts = {name: len(paths) for name, paths in all_class_paths.items()} # Create a dictionary mapping class names to their total image count.
		sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True) # Sort the classes by count in descending order.
		selected_classes = [name for name, count in sorted_classes[:top_n]] # Take the top N classes from the sorted list.
	else: # If the selection mode is unknown.
		raise ValueError(f"Unknown selection_mode: '{selection_mode}'.") # Raise an error.

	if not selected_classes: # If, after filtering, no classes were selected.
		print("❌ Error: No classes were selected.") # Print an error message.
		return [], [], {}, None # Return empty objects to signal failure.

	unified_class_map = {name: i for i, name in enumerate(sorted(selected_classes))} # Create a mapping from the final sorted class names to integer labels (0, 1, 2, ...).
	print("-" * 30) # Print a separator line for visual clarity.
	print(f"✅ Using {len(unified_class_map)} classes: {list(unified_class_map.keys())}") # Print the number and names of the final classes being used.
	print("-" * 30) # Print another separator line.

	# --- 3. Create the final sample lists for the SELECTED classes ---
	final_train_samples = [] # Initialize an empty list for the final training samples.
	final_valid_samples = [] # Initialize an empty list for the final validation samples.

	for class_name, paths in train_samples_map.items(): # Loop through the training map.
		if class_name in unified_class_map: # Check if this class is one of the selected classes.
			for path in paths: # If it is, loop through all its image paths.
				final_train_samples.append((path, unified_class_map[class_name])) # Append a (path, label) tuple to the final list.

	for class_name, paths in valid_samples_map.items(): # Loop through the validation map.
		if class_name in unified_class_map: # Check if this class is one of the selected classes.
			for path in paths: # If it is, loop through all its image paths.
				final_valid_samples.append((path, unified_class_map[class_name])) # Append a (path, label) tuple to the final list.

	print(f"✅ Fused {len(final_train_samples)} training images and {len(final_valid_samples)} validation images.") # Print a summary of the final dataset sizes.
	# Return None for weights, as they will be calculated in the main script
	# based on the final training set composition.
	return final_train_samples, final_valid_samples, unified_class_map, None # Return the final lists and the class map.


class FusedDataset(Dataset):
	"""
	A custom dataset that works with a list of (image_path, label) tuples.
	Includes an option to preload all images into RAM to speed up training.
	"""
	def __init__(self, samples, transform=None, preload=False, safety_margin=0.75):
		"""
		Initializes the dataset.

		Args:
			samples (list): A list of (image_path, label) tuples.
			transform (callable, optional): A function/transform to apply to the images.
			preload (bool): If True, attempts to load all provided samples into memory.
			safety_margin (float): The fraction of available RAM to consider safe for pre-loading.
		"""
		def _is_safe_to_preload(samples_to_check, margin):
			"""Checks if there is enough available RAM to safely preload the dataset."""
			if not PSUTIL_AVAILABLE: # If the psutil library is not available.
				print("\n⚠️ Warning: `psutil` not found. Cannot perform memory safety check. Disabling pre-loading.") # Print a warning.
				print("   Install it with: pip install psutil") # Instruct the user how to install it.
				return False # Return False to disable pre-loading.
			try: # Use a try-except block to catch any potential errors during the check.
				# Estimate dataset size by summing up the file sizes on disk.
				# 'path' items are strings now.
				total_size_bytes = sum(os.path.getsize(path) for path, _ in samples_to_check) # Calculate the total disk size of all image files.
				available_ram_bytes = psutil.virtual_memory().available # Get the amount of currently available system RAM in bytes.

				# Check if the estimated size is within the safe limit of available RAM.
				if total_size_bytes < available_ram_bytes * margin: # Compare the dataset size to a fraction of the available RAM.
					print(f"✅ Memory check passed. Dataset size (~{total_size_bytes / 1e9:.2f} GB) is within safe limits.") # If safe, print a success message.
					return True # Return True to enable pre-loading.
				else: # If the dataset is too large.
					print("\n❌ Memory safety check failed! Pre-loading aborted.") # Print a failure message.
					print(f"   - Estimated dataset size: {total_size_bytes / 1e9:.2f} GB") # Show the estimated size.
					print(f"   - Available RAM: {available_ram_bytes / 1e9:.2f} GB") # Show the available RAM.
					print("   - To prevent system instability, training will proceed by loading images from disk.\n") # Explain the fallback action.
					return False # Return False to disable pre-loading.
			except Exception as e: # If any other error occurs during the check.
				print(f"⚠️ Warning: Could not perform memory safety check. Error: {e}. Disabling pre-loading.") # Print a warning with the error.
				return False # Return False to disable pre-loading.

		self.transform = transform # Store the image transformation pipeline.
		self.samples_are_preloaded = False # A flag to indicate if the samples are in RAM.

		# If preloading is requested for this specific set of samples.
		if preload:
			# Perform safety check before attempting to preload.
			if _is_safe_to_preload(samples, safety_margin):
				print(f"🚀 Pre-loading {len(samples)} images into RAM. This may take a moment...") # Announce the pre-loading process.
				# self.samples will store (PIL.Image, label) tuples instead of (path, label).
				loaded_samples = [] # Initialize an empty list to store the in-memory images and labels.
				for path, label in tqdm(samples, desc="Pre-loading"): # Loop through the samples with a progress bar.
					try: # Use a try-except block to handle corrupted image files.
						# Open the image from the path and convert it to RGB.
						image = Image.open(path).convert("RGB") # Open the image file and ensure it's in RGB format.
						loaded_samples.append((image, label)) # Append the PIL Image object and its label to the list.
					except Exception as e: # If opening the image fails.
						# If an image is corrupted, it will be skipped during pre-loading.
						print(f"⚠️ Warning: Could not load image {path}. Error: {e}. Skipping.") # Print a warning with the file path and error.
				self.samples = loaded_samples # Replace the path list with the list of loaded images.
				self.samples_are_preloaded = True # Set the flag to indicate success.
			else:
				# If safety check fails, fall back to on-demand loading for this set of samples.
				self.samples = samples
		else: # If pre-loading is not requested.
			# If not preloading, just store the list of file paths.
			self.samples = samples # Simply store the original list of (path, label) tuples.

	def __len__(self):
		"""Returns the total number of samples in the dataset."""
		return len(self.samples) # Return the total count of samples.

	def __getitem__(self, idx):
		"""
		Retrieves a sample from the dataset at the given index.
		"""
		if self.samples_are_preloaded: # If the data is pre-loaded in RAM.
			# If preloaded, the image is already a PIL object in memory.
			image, label = self.samples[idx] # Directly get the PIL Image object and label from the list.
		else: # If data is not pre-loaded.
			# If not preloaded, open the image from disk.
			image_path, label = self.samples[idx] # Get the file path and label from the list.
			try: # Use a try-except block to handle potential file reading errors.
				image = Image.open(image_path).convert("RGB") # Open the image from the path and convert to RGB.
			except Exception as e: # If an error occurs.
				print(f"⚠️ Warning: Could not load image {image_path}. Error: {e}. Skipping.") # Print a warning.
				return self.__getitem__((idx + 1) % len(self)) # Skip the bad sample by recursively calling __getitem__ for the next sample.

		# Apply the transformations if they are defined.
		if self.transform: # Check if a transform pipeline was provided.
			image = self.transform(image) # Apply the transformations to the image.

		return image, label # Return the processed image and its label.

# =========================================
# Test Block
# =========================================
if __name__ == '__main__':
	"""
	This block runs only when the script is executed directly.
	It's used for testing the dataset preparation logic in isolation.
	"""
	print("\n--- Running dataset_utils.py in test mode ---")
	
	# Define the path to your datasets for testing.
	# Make sure this path is correct.
	test_dataset_root = r"D:\dev\sih\datasets"
	
	# Call the main function with some test parameters.
	train_s, valid_s, class_map, _ = prepare_fused_samples(
		dataset_dirs=[test_dataset_root],
		selection_mode="TOP_N",
		top_n=5 # Let's just look for the top 5 classes for a quick test.
	)
	
	# Print a summary of the results.
	print("\n--- Test Summary ---")
	if train_s:
		print(f"✅ Successfully found {len(train_s)} training samples.")
		print(f"✅ Successfully found {len(valid_s)} validation samples.")
		print(f"✅ Class map created for {len(class_map)} classes: {list(class_map.keys())}")
	else:
		print("❌ Test failed: No training samples were found.")
		print("Please check the following:")
		print(f"1. The test_dataset_root path is correct: '{test_dataset_root}'")
		print("2. The directory contains subfolders for each class.")
		print("3. The class subfolders contain image files.")