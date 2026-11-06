"""
fast_parallel_download.py

Resilient parallel chunk downloader with retry logic and state persistence.
Downloads only missing chunks and resumes automatically.
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "https://huggingface.co/mergekit-community/Qwen3-1.5B-Instruct/resolve/main/model-00001-of-00001.safetensors"
OUT_DIR = os.path.expanduser("~/.cache/huggingface/hub/models--mergekit-community--Qwen3-1.5B-Instruct/snapshots/675e6f455dc5d0054e78732e855af7b298def5c3")
OUT_FILE = os.path.join(OUT_DIR, "model-00001-of-00001.safetensors")
PART_FILE = OUT_FILE + ".part"
STATE_FILE = OUT_FILE + ".done_chunks"

TOTAL_SIZE = 3087466808  # ~2.88 GiB
NUM_CHUNKS = 32
CHUNK_SIZE = (TOTAL_SIZE + NUM_CHUNKS - 1) // NUM_CHUNKS
MAX_WORKERS = 8

def get_completed_chunks():
    if not os.path.exists(STATE_FILE):
        return set()
    with open(STATE_FILE, "r") as f:
        return {int(line.strip()) for line in f if line.strip().isdigit()}

def mark_chunk_done(chunk_id):
    with open(STATE_FILE, "a") as f:
        f.write(f"{chunk_id}\n")

def download_chunk(chunk_id, start_byte, end_byte):
    for attempt in range(6):
        try:
            req = urllib.request.Request(
                URL,
                headers={
                    "Range": f"bytes={start_byte}-{end_byte}",
                    "User-Agent": "Mozilla/5.0"
                }
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = resp.read()

            if len(data) != (end_byte - start_byte + 1):
                raise ValueError(f"Expected {end_byte - start_byte + 1} bytes, got {len(data)}")

            with open(PART_FILE, "r+b") as f:
                f.seek(start_byte)
                f.write(data)

            mark_chunk_done(chunk_id)
            print(f"[Chunk {chunk_id+1}/{NUM_CHUNKS} COMPLETE] ({len(data)/(1024*1024):.1f} MB)", flush=True)
            return True
        except Exception as e:
            wait_s = (attempt + 1) * 2
            print(f"Warning: Chunk {chunk_id+1} attempt {attempt+1} failed ({e}). Retrying in {wait_s}s...", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"Chunk {chunk_id+1} failed after 6 attempts!")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(OUT_FILE) and os.path.getsize(OUT_FILE) == TOTAL_SIZE:
        print(f"File already complete: {OUT_FILE}")
        return

    # Initialize file if not exists
    if not os.path.exists(PART_FILE) or os.path.getsize(PART_FILE) != TOTAL_SIZE:
        print(f"Pre-allocating {TOTAL_SIZE / (1024**3):.2f} GiB file...")
        with open(PART_FILE, "wb") as f:
            f.truncate(TOTAL_SIZE)

    done = get_completed_chunks()
    # Populate known completed chunks from previous run (26 chunks)
    # [1, 2, 5, 6, 8, 9, 10, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] (1-indexed)
    known_done = {0, 1, 4, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}
    done.update(known_done)
    with open(STATE_FILE, "w") as f:
        for cid in sorted(done):
            f.write(f"{cid}\n")

    chunks_to_download = []
    for i in range(NUM_CHUNKS):
        if i in done:
            continue
        start = i * CHUNK_SIZE
        end = min(TOTAL_SIZE - 1, start + CHUNK_SIZE - 1)
        chunks_to_download.append((i, start, end))

    print(f"Status: {len(done)}/{NUM_CHUNKS} chunks already verified ({len(done)/NUM_CHUNKS*100:.1f}%).")
    print(f"Downloading remaining {len(chunks_to_download)} chunks...")

    if chunks_to_download:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(download_chunk, cid, s, e) for cid, s, e in chunks_to_download]
            for f in as_completed(futures):
                f.result()
        print(f"Remaining chunks downloaded in {time.time()-t0:.1f}s!")

    # Verify size
    if os.path.getsize(PART_FILE) == TOTAL_SIZE:
        os.rename(PART_FILE, OUT_FILE)
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)
        print(f"\nALL 32 CHUNKS ASSEMBLED SUCCESSFULLY!")
        print(f"Saved complete model to: {OUT_FILE}")
    else:
        raise ValueError(f"Size mismatch: {os.path.getsize(PART_FILE)} != {TOTAL_SIZE}")

if __name__ == "__main__":
    main()
