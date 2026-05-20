from __future__ import annotations

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
from safetensors.torch import load_file

from lora_finetuning.common import (apply_config_patch, apply_multi_speaker_patches,
                                    apply_speaker_patch, apply_single_speaker_from_multi,
                                    load_json, load_lora_adapter,
                                    parse_torch_dtype)
from qwen_tts import Qwen3TTSModel


def is_multi_speaker_patch(patch_file: Path) -> bool:
    state = load_file(str(patch_file))
    return any(k.startswith("embedding_") for k in state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS custom voice inference with base model + LoRA bundle")
    parser.add_argument("--bundle_dir", type=str, default=None, help="Bundle dir created by export_custom_voice.py")
    parser.add_argument("--base_model", type=str, default=None, help="Optional override for base model path")
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--config_patch_file", type=str, default=None)
    parser.add_argument("--speaker_patch_file", type=str, default=None)
    parser.add_argument("--speaker_name", type=str, default=None)

    parser.add_argument("--text", required=True)
    parser.add_argument("--output_wav", required=True)
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--device_map", default="cuda:0")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def resolve_bundle_paths(args: argparse.Namespace) -> tuple[str, Path, Path, Path, str | None]:
    if args.bundle_dir:
        bundle_dir = Path(args.bundle_dir)
        manifest = load_json(bundle_dir / "manifest.json")
        base_model = args.base_model or manifest["base_model_path"]
        adapter_dir = Path(args.adapter_dir) if args.adapter_dir else bundle_dir / manifest.get("adapter_dir", "adapter")
        config_patch_file = (
            Path(args.config_patch_file)
            if args.config_patch_file
            else bundle_dir / manifest.get("config_patch_file", "config_patch.json")
        )
        speaker_patch_file = (
            Path(args.speaker_patch_file)
            if args.speaker_patch_file
            else bundle_dir / manifest.get("speaker_embedding_file", "speaker_embedding.safetensors")
        )
        speaker_name = args.speaker_name or manifest.get("speaker_name")
        return base_model, adapter_dir, config_patch_file, speaker_patch_file, speaker_name

    if not all([args.base_model, args.adapter_dir, args.config_patch_file, args.speaker_patch_file]):
        raise ValueError(
            "When --bundle_dir is not provided, you must pass --base_model, --adapter_dir, --config_patch_file and --speaker_patch_file"
        )
    return (
        args.base_model,
        Path(args.adapter_dir),
        Path(args.config_patch_file),
        Path(args.speaker_patch_file),
        args.speaker_name,
    )


def main() -> None:
    args = parse_args()
    base_model, adapter_dir, config_patch_file, speaker_patch_file, speaker_name = resolve_bundle_paths(args)
    torch_dtype = parse_torch_dtype(args.torch_dtype)

    qwen3tts = Qwen3TTSModel.from_pretrained(
        base_model,
        device_map=args.device_map,
        dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=args.local_files_only,
    )

    load_lora_adapter(qwen3tts.model, adapter_dir)
    config_patch = load_json(config_patch_file)
    apply_config_patch(qwen3tts.model, config_patch)

    if is_multi_speaker_patch(speaker_patch_file):
        available_speakers = list(config_patch.get("talker_config", {}).get("spk_id", {}).keys())
        if not available_speakers:
            raise ValueError("Multi-speaker patch but no speakers in config_patch")
        if speaker_name is None:
            if len(available_speakers) > 1:
                raise ValueError(
                    f"Multi-speaker bundle with {len(available_speakers)} speakers. "
                    f"Please specify --speaker_name from: {available_speakers}"
                )
            speaker_name = available_speakers[0]
        if speaker_name not in available_speakers:
            raise ValueError(
                f"Speaker '{speaker_name}' not found. Available: {available_speakers}"
            )
        apply_single_speaker_from_multi(qwen3tts.model, speaker_patch_file, config_patch, speaker_name)
        print(f"Applied multi-speaker patch, selected: {speaker_name}")
    else:
        apply_speaker_patch(qwen3tts.model, speaker_patch_file)
        if speaker_name is None:
            speaker_name = next(iter(config_patch["talker_config"]["spk_id"].keys()))

    qwen3tts.model.eval()

    wavs, sample_rate = qwen3tts.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=speaker_name,
        instruct=args.instruct or None,
    )
    output_path = Path(args.output_wav)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), wavs[0], sample_rate)

    print(f"Inference completed: {output_path}")
    print(f"Speaker: {speaker_name}")
    print(f"Device: {args.device_map}")
    print(f"CUDA available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()
