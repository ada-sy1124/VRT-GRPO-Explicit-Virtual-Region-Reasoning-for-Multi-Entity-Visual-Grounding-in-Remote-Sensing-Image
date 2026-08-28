"""Download and extract ME-RSRG archive files from Hugging Face."""

import os

# Use the Hugging Face mirror endpoint when needed.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import HfApi, hf_hub_download
import zipfile
import tarfile

# Paths
REPO_ID = "AlleyOop26/me-rsrg"
SAVE_DIR = "/root/autodl-tmp/SPIN/"
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
os.makedirs(IMAGE_DIR, exist_ok=True)

print(f"Listing dataset repository files: {REPO_ID} ...")
api = HfApi(endpoint="https://hf-mirror.com")

try:
    files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
except Exception as e:
    print(f"Failed to list repository files. Please check the network: {e}")
    exit()

print("\nRepository files:")
for f in files:
    print(f" - {f}")

# Find archive files automatically.
archives = [f for f in files if f.endswith(('.zip', '.tar', '.tar.gz', '.rar', '.7z'))]

if not archives:
    print("\nNo archive files were found in the dataset repository.")
else:
    print(f"\nArchive files found: {archives}")
    for archive in archives:
        print(f"\nDownloading {archive} ...")
        
        # Download archive.
        local_path = hf_hub_download(
            repo_id=REPO_ID, 
            filename=archive, 
            repo_type="dataset", 
            local_dir=SAVE_DIR
        )
        print(f"Downloaded archive to: {local_path}")
        
        # Extract archive.
        print(f"Extracting to: {IMAGE_DIR} ...")
        try:
            if archive.endswith('.zip'):
                with zipfile.ZipFile(local_path, 'r') as zip_ref:
                    zip_ref.extractall(IMAGE_DIR)
            elif archive.endswith(('.tar', '.tar.gz')):
                with tarfile.open(local_path, 'r:*') as tar_ref:
                    tar_ref.extractall(IMAGE_DIR)
            print("Extraction completed.")
        except Exception as e:
            print(f"Archive extraction failed; the archive may be corrupted: {e}")


# python code/tools/download_me_rsrg_archives.py
