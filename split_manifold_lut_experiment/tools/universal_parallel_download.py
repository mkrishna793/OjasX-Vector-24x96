"""
universal_parallel_download.py

Fast multi-stream parallel chunk downloader for any HuggingFace safetensors model.
"""

import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_model(repo_id: str, filename: str = "model.safetensors", max_workers: int = 12):
    url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
    print(f"Connecting to {url}...")

    # Get total size
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        total_size = int(resp.headers.get("Content-Length"))

    model_dir_name = "models--" + repo_id.replace("/", "--")
    out_dir = os.path.expanduser(f"~/.cache/huggingface/hub/{model_dir_name}/snapshots/official_weights")
    os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, filename)
    part_file = out_file + ".part"
    state_file = out_file + ".done"

    print(f"Model: {repo_id} | Total Size: {total_size / (1024**3):.2f} GiB ({total_size:,} bytes)")

    if os.path.exists(out_file) and os.path.getsize(out_file) == total_size:
        print(f"File already completely downloaded at: {out_file}")
        return out_dir

    num_chunks = 24
    chunk_size = (total_size + num_chunks - 1) // num_chunks

    # Pre-allocate
    if not os.path.exists(part_file) or os.path.getsize(part_file) != total_size:
        with open(part_file, "wb") as f:
            f.truncate(total_size)

    done_chunks = set()
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            done_chunks = {int(x.strip()) for x in f if x.strip().isdigit()}

    def download_chunk(cid, start, end):
        for attempt in range(5):
            try:
                r = urllib.request.Request(
                    url,
                    headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(r, timeout=60) as resp:
                    data = resp.read()
                if len(data) != (end - start + 1):
                    raise ValueError("Incomplete chunk read")
                with open(part_file, "r+b") as f:
                    f.seek(start)
                    f.write(data)
                with open(state_file, "a") as sf:
                    sf.write(f"{cid}\n")
                print(f"[{cid+1}/{num_chunks}] Chunk complete ({len(data)/(1024*1024):.1f} MB)", flush=True)
                return True
            except Exception as e:
                time.sleep((attempt + 1) * 2)
        raise RuntimeError(f"Chunk {cid} failed after retries")

    chunks_to_dl = []
    for i in range(num_chunks):
        if i in done_chunks:
            continue
        s = i * chunk_size
        e = min(total_size - 1, s + chunk_size - 1)
        chunks_to_dl.append((i, s, e))

    print(f"Starting parallel download with {max_workers} streams ({len(chunks_to_dl)} chunks to fetch)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_chunk, cid, s, e) for cid, s, e in chunks_to_dl]
        for f in as_completed(futures):
            f.result()

    elapsed = time.time() - t0
    avg_speed = (total_size / (1024 * 1024)) / max(0.1, elapsed)
    print(f"Download completed in {elapsed:.1f}s! ({avg_speed:.1f} MB/s)")

    os.rename(part_file, out_file)
    if os.path.exists(state_file):
        os.remove(state_file)
    print(f"Saved complete weights to: {out_file}")

    # Also download tokenizer and config
    from huggingface_hub import hf_hub_download
    for extra in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]:
        try:
            p = hf_hub_download(repo_id=repo_id, filename=extra, local_dir=out_dir)
            print(f"Downloaded meta: {extra}")
        except Exception as e:
            pass

    return out_dir

if __name__ == "__main__":
    repo = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
    download_model(repo)
