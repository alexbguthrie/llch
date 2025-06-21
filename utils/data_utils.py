import os
import requests
import tarfile
import zipfile
import gzip
import shutil
from tqdm import tqdm
from datasets import load_dataset

def download_file(url, destination):
    """
    Download a file from a URL to a destination
    
    Args:
        url: URL to download from
        destination: Local path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    # Download with progress bar
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file from {url}: {e}")
        # Clean up partially downloaded file if it exists
        if os.path.exists(destination):
            os.remove(destination)
        raise

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Downloading {os.path.basename(destination)}") as pbar:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                f.write(data)
                
    return destination

def extract_tar(tar_path, extract_path=None):
    """
    Extract a tar file
    
    Args:
        tar_path: Path to the tar file
        extract_path: Path to extract to (defaults to the directory containing the tar)
    """
    if extract_path is None:
        extract_path = os.path.dirname(tar_path)
        
    os.makedirs(extract_path, exist_ok=True)
    
    with tarfile.open(tar_path) as tar:
        # Get total size for progress bar
        total_size = sum(member.size for member in tar.getmembers())
        extracted_size = 0
        
        # Extract with progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Extracting {os.path.basename(tar_path)}") as pbar:
            for member in tar.getmembers():
                tar.extract(member, path=extract_path)
                extracted_size += member.size
                pbar.update(member.size)
                
    return extract_path

def extract_zip(zip_path, extract_path=None):
    """
    Extract a zip file
    
    Args:
        zip_path: Path to the zip file
        extract_path: Path to extract to (defaults to the directory containing the zip)
    """
    if extract_path is None:
        extract_path = os.path.dirname(zip_path)
        
    os.makedirs(extract_path, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get total size for progress bar
        total_size = sum(info.file_size for info in zip_ref.infolist())
        extracted_size = 0
        
        # Extract with progress bar
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Extracting {os.path.basename(zip_path)}") as pbar:
            for info in zip_ref.infolist():
                zip_ref.extract(info, path=extract_path)
                extracted_size += info.file_size
                pbar.update(info.file_size)
                
    return extract_path

def extract_gzip(gzip_path, extract_path=None):
    """
    Extract a gzip file
    
    Args:
        gzip_path: Path to the gzip file
        extract_path: Path to extract to (defaults to the same name without .gz)
    """
    if extract_path is None:
        extract_path = gzip_path.replace('.gz', '')
        
    with gzip.open(gzip_path, 'rb') as f_in:
        with open(extract_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
            
    return extract_path

def download_wikitext(data_dir='data/wikitext-2'):
    """
    Download and extract the WikiText-2 dataset using the datasets library.
    
    Args:
        data_dir: Directory to save the dataset
    """
    # Create the target directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    try:
        # Use the datasets library to download and load the dataset
        print("Downloading WikiText-2 dataset from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        
        # Save the splits to our local data directory
        split_map = {'train': 'train', 'validation': 'valid', 'test': 'test'}
        for hf_split, local_split in split_map.items():
            output_path = os.path.join(data_dir, f"wiki.{local_split}.tokens")
            
            if not os.path.exists(output_path):
                print(f"Saving {hf_split} split to {output_path}...")
                with open(output_path, 'w', encoding='utf-8') as f:
                    for example in dataset[hf_split]:
                        # Write non-empty text, ensuring a newline
                        text = example['text'].strip()
                        if text:
                            f.write(text + '\n')

    except Exception as e:
        print(f"An error occurred while downloading the dataset: {e}")
        print("Please check your internet connection and Hugging Face access.")
        raise
        
    return data_dir

def download_bookcorpus(data_dir='data/bookcorpus'):
    """
    Download and extract a sample of the BookCorpus dataset
    
    Note: The full BookCorpus is no longer publicly available.
    This function downloads a small sample for demonstration purposes.
    
    Args:
        data_dir: Directory to save the dataset
    """
    # This is a placeholder URL - you would need to replace with a real source
    url = "https://huggingface.co/datasets/bookcorpus/resolve/main/data/train-00000-of-00017.parquet"
    parquet_path = os.path.join(data_dir, "bookcorpus-sample.parquet")
    
    # Download if not already downloaded
    if not os.path.exists(parquet_path):
        os.makedirs(data_dir, exist_ok=True)
        try:
            download_file(url, parquet_path)
        except Exception as e:
            print(f"Error downloading BookCorpus: {e}")
            print("Note: The full BookCorpus dataset is no longer publicly available.")
            print("You may need to use an alternative dataset or find a different source.")
            return None
            
    return data_dir

def preprocess_text(text):
    """
    Preprocess text for training
    
    Args:
        text: Input text
    
    Returns:
        Preprocessed text
    """
    # Simple preprocessing steps
    # 1. Convert to lowercase
    text = text.lower()
    
    # 2. Replace multiple spaces with a single space
    text = ' '.join(text.split())
    
    # 3. Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def load_text_files(directory, extension='.txt'):
    """
    Load all text files from a directory
    
    Args:
        directory: Directory containing text files
        extension: File extension to look for
        
    Returns:
        List of text contents
    """
    texts = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(preprocess_text(text))
                
    return texts 