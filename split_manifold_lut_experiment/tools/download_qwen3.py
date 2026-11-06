"""
download_qwen3.py

Downloads mergekit-community/Qwen3-1.5B-Instruct directly from HuggingFace
to local cache with progress reporting.
"""

import sys
import time
from huggingface_hub import hf_hub_download

MODEL_ID = "mergekit-community/Qwen3-1.5B-Instruct"
FILENAME = "model-00001-of-00001.safetensors"

def main():
    print(f"Starting download of {MODEL_ID}/{FILENAME} (~3.08 GB)...")
    t0 = time.time()
    try:
        path = hf_hub_download(
            repo_id=MODEL_ID,
            filename=FILENAME,
            resume_download=True
        )
        elapsed = time.time() - t0
        print(f"\nDownload completed in {elapsed:.1f}s!")
        print(f"File stored at: {path}")
    except Exception as e:
        print(f"Error during download: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
