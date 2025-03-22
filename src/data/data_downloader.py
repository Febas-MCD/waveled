import os
import requests
import shutil # For moving files
import platform  # For checking the operating system
import subprocess  # For calling the tar command
from tqdm import tqdm  # For progress bar


# Define the base directories
base_dirs = ['interim', 'processed', 'raw']

# Define the subdirectories to create in each base directory
sub_dirs = ['bal_train', 'eval', 'unbal_train']

def create_folders(base_path, sub_dirs):
    """
    Create subdirectories inside a base directory.
    
    Args:
        base_path (str): The path to the base directory.
        sub_dirs (list): List of subdirectories to create.
    """
    for sub_dir in sub_dirs:
        dir_path = os.path.join(base_path, sub_dir)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

# Define the files to download and their save paths
files_to_download = {
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/eval_segments.csv": "data/raw/eval/eval_segments.csv",
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/balanced_train_segments.csv": "data/raw/bal_train/balanced_train_segments.csv",
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv": "data/raw/unbal_train/unbalanced_train_segments.csv",
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv": "data/raw/class_labels_indices.csv",
    "https://raw.githubusercontent.com/audioset/ontology/d417d32bf59c711abb5910fd2f76a0eb44697991/ontology.json": "data/raw/ontology.json"
}

# Define the .tar.gz file to download
tar_gz_url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz"
tar_gz_path = "data/raw/features.tar.gz"

def download_file(url, save_path):
    """
    Download a file from a URL and save it to the specified path with a progress bar.
    
    Args:
        url (str): The URL of the file to download.
        save_path (str): The path to save the downloaded file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print(f"Downloading {url} to {save_path}...")
    
    # Stream the download and use tqdm for progress bar
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get("content-length", 0))  # Total file size in bytes
        block_size = 8192  # Chunk size for downloading
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(save_path))
        
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                progress_bar.update(len(chunk))  # Update progress bar
        progress_bar.close()
        
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def move_and_cleanup(extract_to):
    """
    Move the contents of the 'audioset_v1_embeddings' folder to the 'raw' directory
    and delete the now-empty 'audioset_v1_embeddings' folder.
    
    Args:
        extract_to (str): Directory where the archive was extracted.
    """
    # Path to the 'audioset_v1_embeddings' folder
    embeddings_folder = os.path.join(extract_to, "audioset_v1_embeddings")
    
    if os.path.exists(embeddings_folder):
        print(f"Moving contents of {embeddings_folder} to {extract_to}...")
        
        # Move all files and subdirectories from 'audioset_v1_embeddings' to 'raw'
        for item in os.listdir(embeddings_folder):
            src = os.path.join(embeddings_folder, item)
            dst = os.path.join(extract_to, item)
            shutil.move(src, dst)
            print(f"Moved {src} to {dst}")
        
        # Delete the now-empty 'audioset_v1_embeddings' folder
        os.rmdir(embeddings_folder)
        print(f"Deleted empty folder: {embeddings_folder}")
    else:
        print(f"Folder {embeddings_folder} does not exist.")

def delete_file(file_path):
    """
    Delete a file if it exists.
    
    Args:
        file_path (str): Path to the file to delete.
    """
    if os.path.exists(file_path):
        print(f"Deleting {file_path}...")
        os.remove(file_path)
        print(f"Deleted {file_path}")
    else:
        print(f"{file_path} does not exist.")

def extract_with_tar_and_clean(archive_path, extract_to):
    """
    Extract a .tar.gz file using the `tar` command on Linux.
    On Windows, print a message instructing the user to extract manually.
    
    Args:
        archive_path (str): Path to the .tar.gz file.
        extract_to (str): Directory to extract the contents to.
    """
    print(f"Extracting {archive_path} to {extract_to}...")
    
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)
    
    # Check the operating system
    if platform.system() == "Linux":
        # Run the tar command on Linux
        command = ["tar", "-xzf", archive_path, "-C", extract_to]
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Extracted {archive_path} to {extract_to}")

            # Move contents of 'audioset_v1_embeddings' to 'raw' and delete the empty folder
            move_and_cleanup("data/raw")

            # Delete the .tar.gz file after extraction
            delete_file(tar_gz_path)
            
        else:
            print(f"Failed to extract {archive_path}. Error: {result.stderr}")
    elif platform.system() == "Windows":
        # Print a message for Windows users
        print("Please extract the file manually using 7-Zip to handle case-sensitive values.")
    else:
        print(f"Unsupported operating system: {platform.system()}")

def main():
    # Download the CSV files
    for url, save_path in files_to_download.items():
        download_file(url, save_path)

    # Download the .tar.gz file
    download_file(tar_gz_url, tar_gz_path)

    # Extract the .tar.gz file using the tar command
    extract_with_tar_and_clean(tar_gz_path, "data/raw")

if __name__ == "__main__":
    main()