from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lora_finetuning.common import (build_bundle_manifest, ensure_dir, load_json,
                                    load_multi_speaker_patch, save_json,
                                    save_speaker_patch)
from safetensors.torch import load_file


def is_multi_speaker_patch(patch_file: Path) -> bool:
    state = load_file(str(patch_file))
    return any(k.startswith("embedding_") for k in state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Qwen3-TTS LoRA artifacts into a reusable custom voice package")
    parser.add_argument("--base_model", required=True, help="Base model repo id or local path used for training")
    parser.add_argument("--source_dir", required=True, help="Training artifact dir, e.g. outputs/lora_single_speaker")
    parser.add_argument("--output_dir", required=True, help="Export bundle directory")
    parser.add_argument("--speaker_name", default=None, help="Which speaker to include (required for multi-speaker source)")
    return parser.parse_args()


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif src.is_file():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = ensure_dir(args.output_dir)

    adapter_dir = source_dir / "adapter"
    config_patch_file = source_dir / "config_patch.json"
    speaker_patch_file = source_dir / "speaker_embedding.safetensors"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")
    if not config_patch_file.exists():
        raise FileNotFoundError(f"Config patch not found: {config_patch_file}")
    if not speaker_patch_file.exists():
        raise FileNotFoundError(f"Speaker patch not found: {speaker_patch_file}")

    config_patch = load_json(config_patch_file)
    spk_id = config_patch.get("talker_config", {}).get("spk_id", {})
    multi_speaker = len(spk_id) > 1

    if multi_speaker and is_multi_speaker_patch(speaker_patch_file):
        if not args.speaker_name:
            available = list(spk_id.keys())
            raise ValueError(
                f"Multi-speaker source with {len(available)} speakers. "
                f"Please specify --speaker_name from: {available}"
            )
        if args.speaker_name not in spk_id:
            available = list(spk_id.keys())
            raise ValueError(
                f"Speaker '{args.speaker_name}' not found. Available: {available}"
            )
        speaker_name = args.speaker_name
        speaker_id = int(spk_id[speaker_name])

        patches = load_multi_speaker_patch(speaker_patch_file)
        if speaker_id not in patches:
            raise FileNotFoundError(
                f"Embedding for speaker '{speaker_name}' (id={speaker_id}) not found in patch file"
            )
        save_speaker_patch(output_dir / "speaker_embedding.safetensors", speaker_id, patches[speaker_id])

        single_config = {
            "tts_model_type": config_patch["tts_model_type"],
            "talker_config": {
                "spk_id": {speaker_name: speaker_id},
                "spk_is_dialect": {speaker_name: config_patch.get("talker_config", {}).get("spk_is_dialect", {}).get(speaker_name, False)},
            },
        }
        save_json(single_config, output_dir / "config_patch.json")

        print(f"Export completed: {output_dir}")
        print(f"Speaker: {speaker_name} (id={speaker_id}) [from multi-speaker source]")
    else:
        speaker_name = args.speaker_name or next(iter(spk_id.keys()))
        speaker_id = int(spk_id[speaker_name])

        copy_if_exists(speaker_patch_file, output_dir / "speaker_embedding.safetensors")
        copy_if_exists(config_patch_file, output_dir / "config_patch.json")
        print(f"Export completed: {output_dir}")
        print(f"Speaker: {speaker_name} (id={speaker_id})")

    copy_if_exists(adapter_dir, output_dir / "adapter")
    copy_if_exists(source_dir / "train_args.json", output_dir / "train_args.json")
    copy_if_exists(source_dir / "metrics.json", output_dir / "metrics.json")

    manifest = build_bundle_manifest(
        base_model_path=args.base_model,
        speaker_name=speaker_name,
        speaker_id=speaker_id,
        adapter_subdir="adapter",
    )
    save_json(manifest, output_dir / "manifest.json")

    print(f"Base model: {args.base_model}")


if __name__ == "__main__":
    main()
