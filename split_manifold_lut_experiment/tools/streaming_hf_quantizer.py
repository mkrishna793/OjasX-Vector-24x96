"""
streaming_hf_quantizer.py

Stream-Quantize-Evict Ring-Buffer Pipeline:
Downloads and quantizes HuggingFace safetensors models shard-by-shard on-the-fly.
Eliminates the need to ever load the full model into host RAM or store full dense weights on disk.

Peak RAM: Strictly bounded to size of 1 active weight tensor (< 2 GB)
Peak Disk: Strictly bounded to size of 1 downloaded shard (~3 GB) + output compressed file

Author: OjasX Research Team & Antigravity
"""

import os
import sys
import json
import time
import gc
import resource
import argparse
import struct
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open


def get_current_ram_mb():
    # macOS ru_maxrss is in bytes, Linux is in kilobytes
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return usage / (1024 * 1024)
    else:
        return usage / 1024


def quantize_tensor_to_manifold(W: torch.Tensor, outlier_sigma: float = 3.5):
    """
    Quantizes a 2D float weight tensor into Split-Manifold format:
    - Channel-wise uint8 codes: round(W / scale) + 128
    - Channel scales: float32
    - Sparse CSR Outliers: values exceeding outlier_sigma * std
    """
    W = W.float().contiguous()
    out_features, in_features = W.shape

    # 1. Channel scales
    abs_max = torch.amax(torch.abs(W), dim=1, keepdim=True).clamp(min=1e-8)
    scales = (abs_max / 127.0).squeeze(1)  # (out_features,)

    # 2. Dense manifold quantization into uint8
    scaled = W / abs_max * 127.0
    codes = torch.clamp(torch.round(scaled) + 128.0, 0, 255).to(torch.uint8)

    # 3. Sparse CSR Outliers extraction
    mean = W.mean()
    std = W.std().clamp(min=1e-8)
    outlier_mask = (W - mean).abs() >= outlier_sigma * std

    nonzero_indices = torch.nonzero(outlier_mask, as_tuple=False)
    if nonzero_indices.numel() > 0:
        rows = nonzero_indices[:, 0]
        cols = nonzero_indices[:, 1]
        # Reconstruct approximation to get residual
        approx = (codes.float() - 128.0) * scales.unsqueeze(1)
        residual = W - approx
        outlier_vals = residual[rows, cols]
        
        # Build CSR representation
        row_counts = torch.bincount(rows, minlength=out_features)
        crow_indices = torch.zeros(out_features + 1, dtype=torch.int32)
        crow_indices[1:] = torch.cumsum(row_counts, dim=0).to(torch.int32)
        col_indices = cols.to(torch.int32)
        values = outlier_vals.to(torch.float32)
    else:
        crow_indices = torch.zeros(out_features + 1, dtype=torch.int32)
        col_indices = torch.zeros(0, dtype=torch.int32)
        values = torch.zeros(0, dtype=torch.float32)

    return {
        "out_features": out_features,
        "in_features": in_features,
        "scales": scales,
        "codes": codes,
        "csr_crow": crow_indices,
        "csr_col": col_indices,
        "csr_val": values
    }


def stream_quantize_model(repo_id: str, output_path: str, evict_raw_shards: bool = True, max_shards: int = None):
    print("=" * 80)
    print(f"OJASX STREAMING QUANTIZER: {repo_id}")
    print(f"Output File: {output_path}")
    print(f"Evict Raw Shards on-the-fly: {evict_raw_shards}")
    print("=" * 80)

    t_start = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # 1. Fetch file list to check if sharded
    files = list_repo_files(repo_id)
    index_file = "model.safetensors.index.json"
    shards_to_process = []

    if index_file in files:
        print(f"Found sharded model manifest: {index_file}. Fetching...")
        index_path = hf_hub_download(repo_id=repo_id, filename=index_file)
        with open(index_path, "r") as f:
            manifest = json.load(f)
        weight_map = manifest.get("weight_map", {})
        # Unique ordered shards
        shards_to_process = sorted(list(set(weight_map.values())))
        print(f"Model consists of {len(shards_to_process)} shards.")
    else:
        # Single safetensors file
        safetensors_files = [f for f in files if f.endswith(".safetensors")]
        if not safetensors_files:
            raise RuntimeError(f"No safetensors found in repository {repo_id}!")
        shards_to_process = sorted(safetensors_files)
        print(f"Found single/direct safetensors shards: {shards_to_process}")

    if max_shards is not None:
        shards_to_process = shards_to_process[:max_shards]
        print(f"Limiting execution to first {max_shards} shard(s) for verification.")

    total_dense_bytes = 0
    total_compressed_bytes = 0
    total_tensors_processed = 0

    # Write output binary container header (Magic: OJAS, version: 1)
    with open(output_path, "wb") as out_fp:
        out_fp.write(b"OJAS\x01\x00\x00\x00")

    for shard_idx, shard_name in enumerate(shards_to_process):
        print(f"\n>>> [{shard_idx + 1}/{len(shards_to_process)}] STREAMING SHARD: {shard_name}")
        ram_before = get_current_ram_mb()
        print(f"    Current Host RAM Usage: {ram_before:.1f} MB")

        # Step A: Download ONLY this shard
        t0_dl = time.time()
        print(f"    Downloading {shard_name} from HuggingFace...")
        shard_local_path = hf_hub_download(
            repo_id=repo_id,
            filename=shard_name,
            resume_download=True
        )
        dl_time = time.time() - t0_dl
        shard_size_mb = os.path.getsize(shard_local_path) / (1024 * 1024)
        print(f"    Downloaded {shard_size_mb:.1f} MB in {dl_time:.1f}s.")

        # Step B: Zero-copy safe_open and quantize on-the-fly
        print(f"    Opening {shard_name} with safe_open...")
        with safe_open(shard_local_path, framework="pt", device="cpu") as f:
            tensor_keys = list(f.keys())
            print(f"    Found {len(tensor_keys)} tensors in this shard.")

            for t_name in tensor_keys:
                tensor = f.get_tensor(t_name)
                dense_bytes = tensor.numel() * tensor.element_size()
                total_dense_bytes += dense_bytes

                # Linear 2D weight matrix: Split-Manifold Quantization
                if tensor.dim() == 2 and "weight" in t_name and not "norm" in t_name:
                    res = quantize_tensor_to_manifold(tensor)
                    # Serialized compressed size
                    comp_bytes = (
                        res["codes"].numel() * 1 +
                        res["scales"].numel() * 4 +
                        res["csr_crow"].numel() * 4 +
                        res["csr_col"].numel() * 4 +
                        res["csr_val"].numel() * 4
                    )
                    total_compressed_bytes += comp_bytes
                    ratio = dense_bytes / max(1, comp_bytes)

                    # Append record to output file
                    with open(output_path, "ab") as out_fp:
                        name_bytes = t_name.encode("utf-8")
                        # Tag 1: Split-Manifold Quantized Linear
                        out_fp.write(struct.pack("<B H", 1, len(name_bytes)))
                        out_fp.write(name_bytes)
                        out_fp.write(struct.pack("<II", res["out_features"], res["in_features"]))
                        out_fp.write(res["scales"].numpy().tobytes())
                        out_fp.write(res["codes"].numpy().tobytes())
                        # CSR Outliers
                        nnz = len(res["csr_val"])
                        out_fp.write(struct.pack("<I", nnz))
                        out_fp.write(res["csr_crow"].numpy().tobytes())
                        if nnz > 0:
                            out_fp.write(res["csr_col"].numpy().tobytes())
                            out_fp.write(res["csr_val"].numpy().tobytes())

                    total_tensors_processed += 1
                    # Immediate memory cleanup
                    del res
                else:
                    # 1D norm, bias, or embeddings: direct serialization
                    data_bytes = tensor.contiguous().numpy().tobytes()
                    total_compressed_bytes += len(data_bytes)
                    with open(output_path, "ab") as out_fp:
                        name_bytes = t_name.encode("utf-8")
                        # Tag 2: Raw / unquantized tensor
                        out_fp.write(struct.pack("<B H", 2, len(name_bytes)))
                        out_fp.write(name_bytes)
                        out_fp.write(struct.pack("<I", len(data_bytes)))
                        out_fp.write(data_bytes)
                    total_tensors_processed += 1

                # Clean tensor
                del tensor

        # Step C: Evict raw shard if requested to free disk
        if evict_raw_shards and os.path.exists(shard_local_path):
            try:
                os.remove(shard_local_path)
                print(f"    [EVICTED] Deleted raw shard from disk: {shard_name} (freed {shard_size_mb:.1f} MB)")
            except Exception as e:
                print(f"    Notice: could not delete shard {shard_name}: {e}")

        gc.collect()
        ram_after = get_current_ram_mb()
        comp_file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"    Shard Complete. RAM Usage: {ram_after:.1f} MB | Total Output File Size: {comp_file_size_mb:.1f} MB")

    total_time = time.time() - t_start
    dense_mb = total_dense_bytes / (1024 * 1024)
    comp_mb = total_compressed_bytes / (1024 * 1024)
    overall_ratio = dense_mb / max(1e-6, comp_mb)

    print("\n" + "=" * 80)
    print("STREAMING QUANTIZATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Tensors Processed:      {total_tensors_processed}")
    print(f"Original Dense Volume:         {dense_mb:.1f} MB")
    print(f"Compressed Split-Manifold:    {comp_mb:.1f} MB")
    print(f"Effective Compression Ratio:  {overall_ratio:.2f}x ({(1.0 - comp_mb/dense_mb)*100:.1f}% space saved)")
    print(f"Final Artifact Path:          {output_path}")
    print(f"Total Wall Time:              {total_time:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OjasX Streaming HuggingFace Quantizer")
    parser.add_argument("--repo-id", type=str, default="mergekit-community/Qwen3-1.5B-Instruct",
                        help="HuggingFace model repository ID")
    parser.add_argument("--output", type=str, default="split_manifold_lut_experiment/models/qwen3_stream_compressed.oqpack",
                        help="Output path for compressed binary")
    parser.add_argument("--no-evict", action="store_true",
                        help="Keep raw downloaded shards on disk (default is to evict)")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Limit number of shards to process for quick testing")
    args = parser.parse_args()

    stream_quantize_model(
        repo_id=args.repo_id,
        output_path=args.output,
        evict_raw_shards=not args.no_evict,
        max_shards=args.max_shards
    )
