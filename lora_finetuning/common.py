from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence, cast

import librosa
import torch
import yaml
from peft import (LoraConfig, TaskType, get_peft_model_state_dict,
                  inject_adapter_in_model, set_peft_model_state_dict)
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALL_LAYER_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "v_proj",
    "o_proj",
]

DEFAULT_TARGET_SCOPE = "talker_only"
SUPPORTED_TARGET_SCOPES = [
    "talker_only",
    "talker_and_code_predictor",
]

PHASE2_OPTIONAL_MODULES = [
    "linear_fc1",
    "linear_fc2",
]


_TARGET_PATH_PATTERNS = {
    "q_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.self_attn\.q_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.self_attn\.q_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.self_attn\.q_proj",
        ],
    },
    "k_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.self_attn\.k_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.self_attn\.k_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.self_attn\.k_proj",
        ],
    },
    "v_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.self_attn\.v_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.self_attn\.v_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.self_attn\.v_proj",
        ],
    },
    "o_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.self_attn\.o_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.self_attn\.o_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.self_attn\.o_proj",
        ],
    },
    "gate_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.mlp\.gate_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.mlp\.gate_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.mlp\.gate_proj",
        ],
    },
    "up_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.mlp\.up_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.mlp\.up_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.mlp\.up_proj",
        ],
    },
    "down_proj": {
        "talker_only": [r"talker\.model\.layers\.\d+\.mlp\.down_proj"],
        "talker_and_code_predictor": [
            r"talker\.model\.layers\.\d+\.mlp\.down_proj",
            r"talker\.code_predictor\.model\.layers\.\d+\.mlp\.down_proj",
        ],
    },
    "linear_fc1": {
        "talker_only": [r"talker\.text_projection\.linear_fc1"],
        "talker_and_code_predictor": [r"talker\.text_projection\.linear_fc1"],
    },
    "linear_fc2": {
        "talker_only": [r"talker\.text_projection\.linear_fc2"],
        "talker_and_code_predictor": [r"talker\.text_projection\.linear_fc2"],
    },
    "small_to_mtp_projection": {
        "talker_only": [r"talker\.code_predictor\.small_to_mtp_projection"],
        "talker_and_code_predictor": [r"talker\.code_predictor\.small_to_mtp_projection"],
    },
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Config file must contain a mapping, got {type(data)!r}")
    return data


def save_json(data: dict[str, Any] | list[Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_setting(
    cli_value: Any,
    config: dict[str, Any],
    section: str | None,
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    if section and isinstance(config.get(section), dict) and key in config[section]:
        return config[section][key]
    if key in config:
        return config[key]
    return default


def normalize_string_list(value: Any, default: Sequence[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise TypeError(f"Expected string or sequence for module list, got {type(value)!r}")


def build_target_module_regex(
    target_modules: Sequence[str] | str | None,
    target_scope: str = DEFAULT_TARGET_SCOPE,
    target_module_regex: str | None = None,
) -> str:
    if target_module_regex:
        return str(target_module_regex).strip()

    if target_scope not in SUPPORTED_TARGET_SCOPES:
        raise ValueError(
            f"Unsupported target_scope: {target_scope}. Expected one of {SUPPORTED_TARGET_SCOPES}."
        )

    normalized_modules = normalize_string_list(target_modules, DEFAULT_TARGET_MODULES)
    if not normalized_modules:
        raise ValueError("At least one target module must be provided for LoRA injection.")

    patterns: list[str] = []
    seen_patterns: set[str] = set()

    for module_name in normalized_modules:
        scoped_patterns = _TARGET_PATH_PATTERNS.get(module_name, {}).get(target_scope)
        if scoped_patterns:
            for pattern in scoped_patterns:
                if pattern not in seen_patterns:
                    patterns.append(pattern)
                    seen_patterns.add(pattern)
            continue

        fallback_pattern = rf".*\.{re.escape(module_name)}"
        if fallback_pattern not in seen_patterns:
            patterns.append(fallback_pattern)
            seen_patterns.add(fallback_pattern)

    return rf"^(?:{'|'.join(patterns)})$"


def parse_torch_dtype(dtype_name: str | None) -> torch.dtype:
    normalized = (dtype_name or "bfloat16").lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[normalized]


def freeze_all_parameters(model: torch.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def mark_trainable_by_name(model: torch.nn.Module, name_fragments: Iterable[str]) -> list[str]:
    enabled: list[str] = []
    fragments = [frag for frag in name_fragments if frag]
    if not fragments:
        return enabled
    for name, param in model.named_parameters():
        if any(fragment in name for fragment in fragments):
            param.requires_grad = True
            enabled.append(name)
    return enabled


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return trainable, total


def build_lora_config(
    r: int,
    alpha: int,
    dropout: float,
    target_modules: Sequence[str] | str,
    bias: str = "none",
) -> LoraConfig:
    bias_value = cast(Literal["none", "all", "lora_only"], bias)
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias=bias_value,
        target_modules=target_modules if isinstance(target_modules, str) else list(target_modules),
        task_type=TaskType.CAUSAL_LM,
    )


def inject_lora(
    model: torch.nn.Module,
    lora_config: LoraConfig,
    extra_trainable_modules: Sequence[str] | None = None,
) -> tuple[int, int, list[str]]:
    freeze_all_parameters(model)
    inject_adapter_in_model(lora_config, model)
    enabled = mark_trainable_by_name(model, extra_trainable_modules or [])
    return (*count_parameters(model), enabled)


def extract_target_speaker_embedding(qwen3tts: Any, ref_audio: str | Path) -> torch.Tensor:
    normalized = qwen3tts._normalize_audio_inputs(str(ref_audio))
    wav, sr = normalized[0]
    target_sr = qwen3tts.model.speaker_encoder_sample_rate
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    speaker_embedding = qwen3tts.model.extract_speaker_embedding(wav, sr).detach().cpu()
    return speaker_embedding


def collect_unique_ref_audios(train_data: list[dict[str, Any]]) -> list[str]:
    paths: set[str] = set()
    for item in train_data:
        ref = item.get("ref_audio", "")
        if ref:
            paths.add(ref)
    return sorted(paths)


def infer_speaker_name_from_ref_path(ref_path: str) -> str:
    stem = Path(ref_path).stem
    for prefix in ("ref_", "reference_", "speaker_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix):]
    return stem


@torch.inference_mode()
def extract_multi_speaker_embeddings(
    qwen3tts: Any,
    ref_audio_paths: list[str],
    base_speaker_id: int = 3000,
    speaker_name_map: dict[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    speakers: dict[str, dict[str, Any]] = {}
    for idx, ref_path in enumerate(ref_audio_paths):
        embedding = extract_target_speaker_embedding(qwen3tts, ref_path)
        if speaker_name_map and ref_path in speaker_name_map:
            name = speaker_name_map[ref_path]
        else:
            name = infer_speaker_name_from_ref_path(ref_path)
        speakers[ref_path] = {
            "speaker_name": name,
            "speaker_id": base_speaker_id + idx,
            "embedding": embedding,
        }
    return speakers


def make_config_patch(speaker_name: str, speaker_id: int) -> dict[str, Any]:
    return {
        "tts_model_type": "custom_voice",
        "talker_config": {
            "spk_id": {
                speaker_name: speaker_id,
            },
            "spk_is_dialect": {
                speaker_name: False,
            },
        },
    }


def make_multi_speaker_config_patch(speakers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    spk_id: dict[str, int] = {}
    spk_is_dialect: dict[str, bool] = {}
    for info in speakers.values():
        spk_id[info["speaker_name"]] = int(info["speaker_id"])
        spk_is_dialect[info["speaker_name"]] = False
    return {
        "tts_model_type": "custom_voice",
        "talker_config": {
            "spk_id": spk_id,
            "spk_is_dialect": spk_is_dialect,
        },
    }


def save_speaker_patch(path: str | Path, speaker_id: int, speaker_embedding: torch.Tensor) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tensor_dict = {
        "speaker_id": torch.tensor([speaker_id], dtype=torch.int64),
        "embedding": speaker_embedding.detach().cpu(),
    }
    save_file(tensor_dict, str(path))


def save_multi_speaker_patch(path: str | Path, speakers: dict[str, dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    tensor_dict: dict[str, torch.Tensor] = {}
    for ref_path, info in speakers.items():
        sid = int(info["speaker_id"])
        tensor_dict[f"embedding_{sid}"] = info["embedding"].detach().cpu()
    save_file(tensor_dict, str(path))


def load_multi_speaker_patch(path: str | Path) -> dict[int, torch.Tensor]:
    state = load_file(str(path))
    patches: dict[int, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("embedding_"):
            sid = int(key.split("_", 1)[1])
            patches[sid] = value
    return patches


def load_speaker_patch(path: str | Path) -> tuple[int, torch.Tensor]:
    state = load_file(str(path))
    if any(k.startswith("embedding_") for k in state):
        multi = load_multi_speaker_patch(path)
        if not multi:
            raise KeyError(f"Empty multi-speaker patch: {path}")
        first_id = next(iter(multi))
        return first_id, multi[first_id]
    if "speaker_id" not in state or "embedding" not in state:
        raise KeyError(f"Speaker patch {path} must contain 'speaker_id' and 'embedding'")
    speaker_id = int(state["speaker_id"].view(-1)[0].item())
    embedding = state["embedding"]
    return speaker_id, embedding


def save_lora_adapter(model: torch.nn.Module, adapter_dir: str | Path, lora_config: LoraConfig) -> Path:
    adapter_dir = ensure_dir(adapter_dir)
    adapter_state = get_peft_model_state_dict(model)
    adapter_state = {key: value.detach().cpu() for key, value in adapter_state.items()}
    save_file(adapter_state, str(adapter_dir / "adapter_model.safetensors"))
    if hasattr(lora_config, "save_pretrained"):
        lora_config.save_pretrained(str(adapter_dir))
    else:
        save_json(lora_config.to_dict(), adapter_dir / "adapter_config.json")
    return adapter_dir


def load_lora_adapter_weights(model: torch.nn.Module, adapter_dir: str | Path) -> LoraConfig:
    adapter_dir = Path(adapter_dir)
    lora_config = cast(LoraConfig, LoraConfig.from_pretrained(str(adapter_dir)))
    adapter_state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    outcome = set_peft_model_state_dict(model, adapter_state)
    unexpected_keys = getattr(outcome, "unexpected_keys", None) if outcome is not None else None
    if unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {unexpected_keys}")
    return lora_config


def load_lora_adapter(model: torch.nn.Module, adapter_dir: str | Path) -> LoraConfig:
    adapter_dir = Path(adapter_dir)
    lora_config = cast(LoraConfig, LoraConfig.from_pretrained(str(adapter_dir)))
    inject_adapter_in_model(lora_config, model)
    load_lora_adapter_weights(model, adapter_dir)
    return lora_config


def apply_config_patch(model: Any, config_patch: dict[str, Any]) -> None:
    tts_model_type = config_patch.get("tts_model_type")
    talker_patch = config_patch.get("talker_config", {})
    spk_id = talker_patch.get("spk_id", {})
    spk_is_dialect = talker_patch.get("spk_is_dialect", {})

    if tts_model_type is not None:
        model.config.tts_model_type = tts_model_type
        model.tts_model_type = tts_model_type

    if spk_id:
        model.config.talker_config.spk_id.update(spk_id)
    if spk_is_dialect:
        model.config.talker_config.spk_is_dialect.update(spk_is_dialect)

    model.supported_speakers = list(model.config.talker_config.spk_id.keys())


def apply_speaker_patch(model: Any, speaker_patch_file: str | Path) -> int:
    speaker_id, embedding = load_speaker_patch(speaker_patch_file)
    target_weight = model.talker.model.codec_embedding.weight
    if speaker_id >= target_weight.shape[0]:
        raise IndexError(
            f"speaker_id {speaker_id} is out of range for codec embedding size {target_weight.shape[0]}"
        )
    embedding = embedding.to(device=target_weight.device, dtype=target_weight.dtype).view(-1)
    if embedding.shape[0] != target_weight.shape[1]:
        raise ValueError(
            f"Speaker embedding dim mismatch: expected {target_weight.shape[1]}, got {embedding.shape[0]}"
        )
    with torch.no_grad():
        target_weight[speaker_id].copy_(embedding)
    return speaker_id


def apply_multi_speaker_patches(
    model: Any,
    speaker_patch_file: str | Path,
    config_patch: dict[str, Any],
    speaker_names: list[str] | None = None,
) -> list[int]:
    patches = load_multi_speaker_patch(speaker_patch_file)
    spk_id = config_patch["talker_config"]["spk_id"]
    target_weight = model.talker.model.codec_embedding.weight

    if speaker_names:
        target_speakers = [s for s in speaker_names if s in spk_id]
    else:
        target_speakers = list(spk_id.keys())

    applied: list[int] = []
    for name in target_speakers:
        sid = int(spk_id[name])
        if sid not in patches:
            continue
        embedding = patches[sid]
        if sid >= target_weight.shape[0]:
            raise IndexError(
                f"speaker_id {sid} for '{name}' is out of range (max {target_weight.shape[0]})"
            )
        embedding = embedding.to(device=target_weight.device, dtype=target_weight.dtype).view(-1)
        if embedding.shape[0] != target_weight.shape[1]:
            raise ValueError(
                f"Speaker embedding dim mismatch for '{name}': "
                f"expected {target_weight.shape[1]}, got {embedding.shape[0]}"
            )
        with torch.no_grad():
            target_weight[sid].copy_(embedding)
        applied.append(sid)

    return applied


def apply_single_speaker_from_multi(
    model: Any,
    speaker_patch_file: str | Path,
    config_patch: dict[str, Any],
    speaker_name: str,
) -> int:
    applied = apply_multi_speaker_patches(model, speaker_patch_file, config_patch, speaker_names=[speaker_name])
    if not applied:
        available = list(config_patch.get("talker_config", {}).get("spk_id", {}).keys())
        raise ValueError(
            f"Speaker '{speaker_name}' not found in multi-speaker patch. Available: {available}"
        )
    return applied[0]


def build_bundle_manifest(
    base_model_path: str,
    speaker_name: str,
    speaker_id: int,
    adapter_subdir: str = "adapter",
    multi_speaker: bool = False,
    speaker_names: list[str] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "format_version": 1,
        "base_model_path": base_model_path,
        "adapter_dir": adapter_subdir,
        "speaker_embedding_file": "speaker_embedding.safetensors",
        "config_patch_file": "config_patch.json",
        "speaker_name": speaker_name,
        "speaker_id": speaker_id,
        "multi_speaker": multi_speaker,
    }
    if multi_speaker and speaker_names:
        manifest["speaker_names"] = speaker_names
    return manifest
