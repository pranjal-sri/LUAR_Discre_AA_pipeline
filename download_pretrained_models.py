import gdown
import os
import subprocess
from pathlib import Path

def download_folder_from_gdrive(folder_url, output_path=None):
    gdown.download_folder(folder_url, output=output_path, quiet=False, use_cookies=False)

def download_file_from_gdrive(file_id, output_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)


def download_sbert_weights(base_dir: str = "./") -> None:
    """
    Downloads the SBERT (Sentence-BERT) weights from HuggingFace using git-lfs.
    
    Args:
        base_dir (str): Base directory where the weights should be downloaded.
                       Defaults to current directory.
    
    Raises:
        subprocess.CalledProcessError: If git-lfs is not installed or if the download fails
        
    Notes:
        - Requires git-lfs to be installed on the system
        - Downloads paraphrase-distilroberta-base-v1 model from HuggingFace
    """
    print("=" * 80)
    print("Make sure you have Git LFS installed & enabled")
    print("Refer to this link for installation instructions: https://git-lfs.github.com")
    print("=" * 80)
    print("\n")
    
    # Create pretrained_weights directory if it doesn't exist
    weights_dir = Path(base_dir) / "pretrained_weights"
    weights_dir.mkdir(exist_ok=True)
    
    # Define model path
    model_name = "paraphrase-distilroberta-base-v1"
    model_path = weights_dir / model_name
    
    # Clone the repository if it doesn't exist
    if not model_path.exists():
        cmd = [
            "git", "clone",
            f"https://huggingface.co/sentence-transformers/{model_name}",
            str(model_path)
        ]
        subprocess.run(cmd, check=True)
    else:
        print(f"Model already exists at {model_path}")

def main():
    file_id = "1_p31FxG0ZwPCQlpOj0tbybxx3U8of9Th"
    output_path = "dummy_texts.csv"
    download_file_from_gdrive(file_id, output_path)

    folder_url = "https://drive.google.com/drive/folders/1o8UUyqkR_YfN_PHKESMJKygBdNpOW_TR"
    download_folder_from_gdrive(folder_url)

    download_sbert_weights()

if __name__ == "__main__":
    main()