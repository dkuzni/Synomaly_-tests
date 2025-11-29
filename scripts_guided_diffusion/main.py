import sys
import os
import argparse
import subprocess
import json
import warnings
import urllib.request
import zipfile
import tarfile

# Temporarily add the current directory to the path to import local modules
sys.path.append(os.getcwd())

# Import necessary files
try:
    from helpers import get_args_from_json
except ImportError:
    warnings.warn("Could not import helpers. Please ensure all files are in the same directory.")
    sys.exit(1)

# --- [1] CONFIGURATION SECTION: SET THESE VARIABLES ---
# -----------------------------------------------------

# REQUIRED: Select the action you want to perform:
# "preprocess": Prepare the raw data (only needs to be run once per raw dataset).
# "train": Train the diffusion model on the healthy data.
# "detect": Run anomaly detection/inference on the test set.
ACTION = "detect"  # Set to "preprocess" for data generation first

# REQUIRED: Select the JSON file (without the '.json' extension) that defines
# the hyperparameters for your desired dataset/model run.
# Make sure this file exists in the 'json_args' directory.
JSON_FILE_NAME = "args_lits"  # <<< CHANGED TO LI.TS CONFIG >>>

# CRITICAL: If a download link is available, paste it here.
# Set to None if you handle the download manually.
DATASET_URL = None  # <<< TODO: PASTE LI.TS DATASET DOWNLOAD URL HERE >>>

# OPTIONAL (For Training): Set a model checkpoint path if resuming training.
# Set to None to start from scratch.
RESUME_MODEL_NAME = None  # e.g., "model/args_us/diff_epoch=100.pt"

# OPTIONAL (For Detection): Set the specific model stage/checkpoint for inference.
MODEL_STAGE = "best_model.pt"  # e.g., "best_model.pt", "params-final.pt", or a specific epoch checkpoint

# OPTIONAL (For Detection): Specify the mode for inference data loading.
# "anomalous" is typically used for calculating metrics.
# "healthy" is for testing model performance on non-anomalous images.
DETECTION_MODE = "anomalous"  # ["anomalous", "healthy", "both"]


# -----------------------------------------------------
# --- [2] EXECUTION LOGIC: DO NOT MODIFY BELOW THIS LINE ---
# -----------------------------------------------------

def download_and_extract_data(url, target_dir):
    """Downloads a file from a URL and extracts it to the target directory."""
    if not url:
        print("\nSkipping automated download: DATASET_URL is not set.")
        return False

    print(f"\n--- Attempting to download data from: {url} ---")
    os.makedirs(target_dir, exist_ok=True)

    filename = url.split('/')[-1]
    download_path = os.path.join(target_dir, filename)

    try:
        # Download the file
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, download_path)
        print(f"Download complete. File saved to {download_path}")

        # Extract the file
        print("Extracting data...")
        if filename.endswith('.zip'):
            with zipfile.ZipFile(download_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        elif filename.endswith('.tar.gz') or filename.endswith('.tgz'):
            with tarfile.open(download_path, 'r:gz') as tar_ref:
                tar_ref.extractall(target_dir)
        else:
            print(f"Warning: File type {filename} not supported for extraction. Please extract manually.")
            return False

        # Clean up the downloaded archive
        os.remove(download_path)
        print("Extraction complete and archive deleted.")
        return True

    except Exception as e:
        print(f"\nFATAL ERROR during data download/extraction: {e}")
        print("Please check the URL or download the data manually and place it in the correct location.")
        return False


def run_script(script_name, args_list=None):
    """Executes a Python script using a subprocess."""
    print(f"\n--- Launching {script_name} ---")

    # Construct the command
    command = [sys.executable, script_name]
    if args_list:
        command.extend(args_list)

    try:
        # Use subprocess.run to execute the script
        # Check=True will raise an error if the script fails
        process = subprocess.run(
            command,
            check=True,
            text=True
        )
        print(f"--- {script_name} completed successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"--- ERROR: {script_name} failed with exit code {e.returncode} ---")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"--- ERROR: Required script '{script_name}' not found. ---")
        sys.exit(1)


def determine_preprocessing_script(dataset_name):
    """Maps dataset name to the correct preprocessing script."""
    dataset_name = dataset_name.lower()
    if dataset_name == "brats23":
        return "data_preprocessing_brats.py"
    elif dataset_name == "lits":
        return "data_preprocessing_lits.py"
    elif dataset_name == "ultrasound":
        return "data_preprocessing_ultrasound.py"
    else:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Check your JSON file.")


def main():
    # 1. Load configuration args (using local server=='None' logic)
    try:
        args = get_args_from_json(JSON_FILE_NAME, server='None')
    except FileNotFoundError:
        print(f"\nFATAL ERROR: JSON config file 'json_args/{JSON_FILE_NAME}.json' not found.")
        print("Please create this file or correct the JSON_FILE_NAME variable.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL ERROR during argument loading: {e}")
        print("Please ensure your dataset name is supported and paths are correct in helpers.py.")
        sys.exit(1)

    dataset = args.get('dataset', 'Unknown')
    raw_data_path = args.get('data_path')

    print("=" * 60)
    print(f"Launching workflow for Dataset: {dataset} | Action: {ACTION.upper()}")
    # Use os.path.abspath for better clarity in the log
    print(f"Raw Data Path: {os.path.abspath(raw_data_path)}")
    print("=" * 60)

    # 2. Check if raw data exists and attempt download if path is missing and URL is provided
    if not os.path.exists(raw_data_path):
        print(f"\nWarning: Raw data directory not found at {raw_data_path}")
        if DATASET_URL:
            # Attempt to download and extract to the parent directory of the raw_data_path
            # e.g., if raw_data_path is '../data/US', the target_dir is '../data'
            parent_dir = os.path.dirname(raw_data_path)

            # If the raw data path is relative to the current directory, the parent is assumed to be the root data folder.
            if parent_dir == "": parent_dir = "."  # Handle cases like 'US' being the path

            success = download_and_extract_data(DATASET_URL, parent_dir)
            if not success and ACTION.lower() == "preprocess":
                print("\nFATAL ERROR: Automated data setup failed. Cannot proceed with preprocessing.")
                sys.exit(1)
        else:
            print("\nFATAL ERROR: Raw data not found and DATASET_URL is missing.")
            print("Please manually download the data and ensure your paths in 'helpers.py' are correct.")
            sys.exit(1)

    # 3. Execute the requested action
    if ACTION.lower() == "preprocess":
        preprocess_script = determine_preprocessing_script(dataset)
        run_script(preprocess_script, args_list=['None'])  # Pass 'None' to use local paths in preprocessing scripts

    elif ACTION.lower() == "train":
        # Launch diffusion_training.py with the JSON file name and optional resume path
        train_args = [JSON_FILE_NAME]
        if RESUME_MODEL_NAME:
            train_args.append(RESUME_MODEL_NAME)

        run_script("diffusion_training.py", args_list=train_args)

    elif ACTION.lower() == "detect":
        # Launch anomaly_detection.py with the JSON file name, mode, and model path
        detect_args = [JSON_FILE_NAME, DETECTION_MODE, MODEL_STAGE]

        run_script("anomaly_detection.py", args_list=detect_args)

    else:
        print(f"ERROR: Invalid ACTION '{ACTION}'. Must be one of 'preprocess', 'train', or 'detect'.")
        sys.exit(1)


if __name__ == '__main__':
    main()