"""Merge 8 band LoRA adapters into a single combined adapter (bangdream-qwen3tts)
using TIES-Merging (Trim, Elect Sign, Disjoint Merge).

Reference: "TIES-Merging: Resolving Interference When Merging Models" (Yadav et al., 2024)

Steps:
  1. Trim — zero out bottom-k% delta values by magnitude (keep top-k%)
  2. Elect Sign — at each parameter position, pick the majority sign across models
  3. Disjoint Merge — average only model deltas whose trimmed sign agrees with the elected sign

Sections 4-6 (speaker embeddings, config) unchanged from simple merge.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SCRIPT_DIR = Path(__file__).resolve().parent
LORA_ROOT = SCRIPT_DIR / "lora"

BAND_NAMES = sorted(
    d.name for d in LORA_ROOT.iterdir()
    if d.is_dir() and d.name.endswith("_qwen3tts")
)

OUTPUT_DIR = LORA_ROOT / "bangdream-qwen3tts"
OUTPUT_ADAPTER_DIR = OUTPUT_DIR / "adapter"


def ties_merge(tensors: torch.Tensor, top_k_frac: float = 0.2) -> torch.Tensor:
    """TIES-Merging over dim=0 (first axis = models).

    Args:
        tensors: shape (num_models, *dims)
        top_k_frac: fraction of largest-magnitude values to keep (default 0.2)

    Returns:
        Merged tensor with same dtype as input.
    """
    dtype = tensors.dtype
    tensors = tensors.float()
    n_models = tensors.shape[0]

    # 1. Trim: per-model, keep top-k% by magnitude
    # Compute threshold separately for each model
    n = tensors[0].numel()
    k = max(1, int(n * top_k_frac))
    flat_abs = tensors.abs().reshape(tensors.shape[0], -1)
    sorted_vals, _ = flat_abs.sort(dim=-1, descending=True)
    # sorted_vals[:, k-1] is the k-th largest value per model
    thresh_idx = min(k, n) - 1
    threshold = sorted_vals[:, thresh_idx]
    # Reshape threshold to broadcast: (num_models, 1, 1, ...)
    threshold = threshold.view(-1, *([1] * (tensors.ndim - 1)))
    trimmed = tensors.where(tensors.abs() >= threshold, torch.zeros_like(tensors))

    # 2. Elect sign: majority sign across models at each position
    sign_sum = trimmed.sum(dim=0)
    elected_sign = sign_sum.sign()
    tie_mask = (sign_sum == 0)  # exact tie — fall back to full average

    # 3. Disjoint merge: average only models agreeing with elected sign
    agree_mask = (trimmed.sign() == elected_sign.unsqueeze(0))
    agree_count = agree_mask.sum(dim=0)
    merged = (trimmed * agree_mask).sum(dim=0) / agree_count.clamp(min=1)
    merged = merged.nan_to_num(0)

    # Tie positions: use simple average (all models contribute equally)
    if tie_mask.any():
        merged[tie_mask] = trimmed[:, tie_mask].mean(dim=0)

    return merged.to(dtype)


def main(top_k: float):
    if not BAND_NAMES:
        print("No band lora directories found!", file=sys.stderr)
        sys.exit(1)

    print(f"TIES-Merging {len(BAND_NAMES)} bands: {BAND_NAMES}")
    print(f"  top_k_frac = {top_k}")

    # ── 1. TIES-Merge adapter weights ──
    print("\n[1/4] TIES-Merging adapter weights...")
    all_adapters = []
    ref_keys = None

    for band in BAND_NAMES:
        adapter_path = LORA_ROOT / band / "adapter" / "adapter_model.safetensors"
        state = load_file(str(adapter_path))
        all_adapters.append(state)
        if ref_keys is None:
            ref_keys = set(state.keys())
        else:
            assert ref_keys == set(state.keys()), f"Key mismatch: {band}"

    keys = sorted(ref_keys)
    merged_adapter = {}
    for key in keys:
        tensors = torch.stack([adapter[key] for adapter in all_adapters])
        merged_adapter[key] = ties_merge(tensors, top_k_frac=top_k)

    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    save_file(merged_adapter, str(OUTPUT_ADAPTER_DIR / "adapter_model.safetensors"))
    print(f"  Saved merged adapter ({len(merged_adapter)} keys)")

    # ── 2. Copy adapter_config.json ──
    print("\n[2/4] Copying adapter_config.json...")
    src_config = LORA_ROOT / BAND_NAMES[0] / "adapter" / "adapter_config.json"
    shutil.copy2(str(src_config), str(OUTPUT_ADAPTER_DIR / "adapter_config.json"))
    print("  Done")

    # ── 3. Merge speaker embeddings ──
    print("\n[3/4] Merging speaker embeddings...")
    merged_spk_id: dict[str, int] = {}
    merged_spk_is_dialect: dict[str, bool] = {}
    merged_embedding: dict[str, torch.Tensor] = {}

    offset = 0
    for band in BAND_NAMES:
        config_path = LORA_ROOT / band / "config_patch.json"
        emb_path = LORA_ROOT / band / "speaker_embedding.safetensors"

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        emb_state = load_file(str(emb_path))

        spk_id_old = config["talker_config"]["spk_id"]
        for name in sorted(spk_id_old.keys(), key=lambda n: spk_id_old[n]):
            old_id = spk_id_old[name]
            new_id = offset
            if name in merged_spk_id:
                print(f"  WARNING: duplicate speaker name '{name}', skipping", file=sys.stderr)
                continue
            merged_spk_id[name] = new_id
            merged_spk_is_dialect[name] = False
            merged_embedding[f"embedding_{new_id}"] = emb_state[f"embedding_{old_id}"]
            offset += 1

    save_file(merged_embedding, str(OUTPUT_DIR / "speaker_embedding.safetensors"))
    print(f"  Total speakers: {len(merged_spk_id)}")
    print(f"  Speaker names: {list(merged_spk_id.keys())}")

    # ── 4. Merge config_patch.json ──
    print("\n[4/4] Writing merged config_patch.json...")
    merged_config = {
        "tts_model_type": "custom_voice",
        "talker_config": {
            "spk_id": merged_spk_id,
            "spk_is_dialect": merged_spk_is_dialect,
        },
    }
    with open(OUTPUT_DIR / "config_patch.json", "w", encoding="utf-8") as f:
        json.dump(merged_config, f, indent=2, ensure_ascii=False)
    print("  Done")

    # ── Verify ──
    print(f"\n=== Merge complete ===")
    print(f"Output: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.rglob("*")):
        if f.is_file():
            size_mb = f.stat().st_size / (1024 * 1024)
            print(f"  {f.relative_to(OUTPUT_DIR)} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIES-Merging for 8 band LoRAs")
    parser.add_argument("--top-k", type=float, default=0.2,
                        help="Top-k fraction to keep during Trim step (default: 0.2)")
    args = parser.parse_args()
    main(top_k=args.top_k)
