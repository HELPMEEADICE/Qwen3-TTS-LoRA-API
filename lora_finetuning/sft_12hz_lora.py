from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, cast

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate import Accelerator
from peft import LoraConfig
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_scheduler

from finetuning.dataset import TTSDataset
from lora_finetuning.common import (DEFAULT_TARGET_MODULES,
                                    DEFAULT_TARGET_SCOPE,
                                    PHASE2_OPTIONAL_MODULES,
                                    SUPPORTED_TARGET_SCOPES,
                                    build_target_module_regex,
                                    build_lora_config,
                                    collect_unique_ref_audios,
                                    ensure_dir,
                                    extract_multi_speaker_embeddings,
                                    extract_target_speaker_embedding,
                                    inject_lora, load_yaml_config,
                                    load_lora_adapter_weights,
                                    make_config_patch,
                                    make_multi_speaker_config_patch,
                                    normalize_string_list,
                                    parse_torch_dtype, resolve_setting,
                                    save_json, save_lora_adapter,
                                    save_multi_speaker_patch,
                                    save_speaker_patch)
from qwen_tts import Qwen3TTSModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-TTS 12Hz Base models")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config file")

    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--init_adapter_dir", type=str, default=None)
    parser.add_argument("--speaker_name", type=str, default=None)
    parser.add_argument("--speaker_id", type=int, default=None)
    parser.add_argument("--speaker_base_id", type=int, default=None)
    parser.add_argument("--speaker_names", nargs="*", default=None)
    parser.add_argument("--multi_speaker", action="store_true", default=None)

    parser.add_argument("--torch_dtype", type=str, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true")

    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--lr_scheduler_type", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--sub_talker_loss_weight", type=float, default=None)
    parser.add_argument("--cpu_threads", type=int, default=None)
    parser.add_argument("--validation_jsonl", type=str, default=None)
    parser.add_argument("--validation_split_ratio", type=float, default=None)
    parser.add_argument("--validation_max_samples", type=int, default=None)
    parser.add_argument("--validation_seed", type=int, default=None)
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)

    parser.add_argument("--lora_r", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lora_dropout", type=float, default=None)
    parser.add_argument("--lora_bias", type=str, default=None)
    parser.add_argument("--target_scope", type=str, choices=SUPPORTED_TARGET_SCOPES, default=None)
    parser.add_argument("--target_module_regex", type=str, default=None)
    parser.add_argument("--target_modules", nargs="*", default=None)
    parser.add_argument("--extra_trainable_modules", nargs="*", default=None)

    return parser.parse_args()


class ResolvedConfig(dict):
    def __getattr__(self, item: str) -> Any:
        return self[item]


def resolve_config(args: argparse.Namespace) -> ResolvedConfig:
    cfg = load_yaml_config(args.config)

    base_model = resolve_setting(args.base_model, cfg, "model", "base_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    train_jsonl = resolve_setting(args.train_jsonl, cfg, "data", "train_jsonl", "train_with_codes.jsonl")
    output_root = resolve_setting(args.output_root, cfg, "artifacts", "output_root", "outputs/lora_single_speaker")
    init_adapter_dir = resolve_setting(args.init_adapter_dir, cfg, "artifacts", "init_adapter_dir", None)
    speaker_name = resolve_setting(args.speaker_name, cfg, "data", "speaker_name", "speaker_test")
    speaker_id = resolve_setting(args.speaker_id, cfg, None, "speaker_id", 3000)
    speaker_base_id = resolve_setting(args.speaker_base_id, cfg, None, "speaker_base_id", 3000)
    speaker_names_override = args.speaker_names if args.speaker_names else (
        resolve_setting(None, cfg, "data", "speaker_names", None)
    )
    multi_speaker_flag = resolve_setting(args.multi_speaker, cfg, "data", "multi_speaker", None)

    target_modules = normalize_string_list(
        args.target_modules if args.target_modules else resolve_setting(None, cfg, "lora", "target_modules", DEFAULT_TARGET_MODULES),
        DEFAULT_TARGET_MODULES,
    )
    target_scope = resolve_setting(args.target_scope, cfg, "lora", "target_scope", DEFAULT_TARGET_SCOPE)
    target_module_regex = resolve_setting(args.target_module_regex, cfg, "lora", "target_module_regex", None)
    target_module_pattern = build_target_module_regex(
        target_modules=target_modules,
        target_scope=target_scope,
        target_module_regex=target_module_regex,
    )
    extra_trainable_modules = normalize_string_list(
        args.extra_trainable_modules if args.extra_trainable_modules else resolve_setting(None, cfg, "lora", "phase2_optional_modules", []),
        [],
    )
    validation_max_samples = resolve_setting(args.validation_max_samples, cfg, "training", "validation_max_samples", None)

    resolved = ResolvedConfig(
        base_model=base_model,
        train_jsonl=train_jsonl,
        validation_jsonl=resolve_setting(args.validation_jsonl, cfg, "data", "validation_jsonl", None),
        output_root=output_root,
        init_adapter_dir=init_adapter_dir,
        speaker_name=speaker_name,
        speaker_id=int(speaker_id),
        speaker_base_id=int(speaker_base_id),
        speaker_names_override=speaker_names_override,
        multi_speaker_flag=multi_speaker_flag,
        torch_dtype=resolve_setting(args.torch_dtype, cfg, "model", "dtype", "bfloat16"),
        attn_implementation=resolve_setting(args.attn_implementation, cfg, "model", "attn_implementation", "sdpa"),
        batch_size=int(resolve_setting(args.batch_size, cfg, "training", "batch_size", 2)),
        gradient_accumulation_steps=int(resolve_setting(args.gradient_accumulation_steps, cfg, "training", "gradient_accumulation_steps", 8)),
        learning_rate=float(resolve_setting(args.learning_rate, cfg, "training", "learning_rate", 5e-5)),
        weight_decay=float(resolve_setting(args.weight_decay, cfg, "training", "weight_decay", 0.0)),
        num_epochs=int(resolve_setting(args.num_epochs, cfg, "training", "num_epochs", 2)),
        warmup_ratio=float(resolve_setting(args.warmup_ratio, cfg, "training", "warmup_ratio", 0.05)),
        lr_scheduler_type=resolve_setting(args.lr_scheduler_type, cfg, "training", "lr_scheduler_type", "linear"),
        max_grad_norm=float(resolve_setting(args.max_grad_norm, cfg, "training", "max_grad_norm", 1.0)),
        mixed_precision=resolve_setting(args.mixed_precision, cfg, "training", "mixed_precision", "bf16"),
        logging_steps=max(1, int(resolve_setting(args.logging_steps, cfg, "training", "logging_steps", 10))),
        sub_talker_loss_weight=float(resolve_setting(args.sub_talker_loss_weight, cfg, None, "sub_talker_loss_weight", 0.3)),
        cpu_threads=resolve_setting(args.cpu_threads, cfg, "training", "cpu_threads", None),
        validation_split_ratio=float(resolve_setting(args.validation_split_ratio, cfg, "training", "validation_split_ratio", 0.0)),
        validation_max_samples=int(validation_max_samples) if validation_max_samples not in (None, "") else None,
        validation_seed=int(resolve_setting(args.validation_seed, cfg, "training", "validation_seed", 42)),
        early_stopping_patience=int(resolve_setting(args.early_stopping_patience, cfg, "training", "early_stopping_patience", 0)),
        early_stopping_min_delta=float(resolve_setting(args.early_stopping_min_delta, cfg, "training", "early_stopping_min_delta", 0.0)),
        lora_r=int(resolve_setting(args.lora_r, cfg, "lora", "r", 16)),
        lora_alpha=int(resolve_setting(args.lora_alpha, cfg, "lora", "alpha", 32)),
        lora_dropout=float(resolve_setting(args.lora_dropout, cfg, "lora", "dropout", 0.05)),
        lora_bias=resolve_setting(args.lora_bias, cfg, "lora", "bias", "none"),
        target_scope=target_scope,
        target_module_regex=target_module_regex,
        target_module_pattern=target_module_pattern,
        target_modules=target_modules,
        extra_trainable_modules=extra_trainable_modules,
        local_files_only=bool(args.local_files_only),
        config_path=args.config,
    )
    return resolved


def _normalize_target_modules_for_compare(target_modules: Sequence[str] | str | None) -> str | tuple[str, ...]:
    if isinstance(target_modules, str):
        return target_modules
    return tuple(normalize_string_list(target_modules, []))


def validate_warm_start_adapter_config(init_adapter_dir: str | Path, expected_lora_config: LoraConfig) -> None:
    adapter_dir = Path(init_adapter_dir)
    stored_lora_config = cast(LoraConfig, LoraConfig.from_pretrained(str(adapter_dir)))

    mismatches: list[str] = []
    if int(stored_lora_config.r) != int(expected_lora_config.r):
        mismatches.append(f"r: expected {expected_lora_config.r}, got {stored_lora_config.r}")
    if int(stored_lora_config.lora_alpha) != int(expected_lora_config.lora_alpha):
        mismatches.append(
            f"lora_alpha: expected {expected_lora_config.lora_alpha}, got {stored_lora_config.lora_alpha}"
        )
    if float(stored_lora_config.lora_dropout) != float(expected_lora_config.lora_dropout):
        mismatches.append(
            f"lora_dropout: expected {expected_lora_config.lora_dropout}, got {stored_lora_config.lora_dropout}"
        )
    if str(stored_lora_config.bias) != str(expected_lora_config.bias):
        mismatches.append(f"bias: expected {expected_lora_config.bias}, got {stored_lora_config.bias}")
    if _normalize_target_modules_for_compare(stored_lora_config.target_modules) != _normalize_target_modules_for_compare(expected_lora_config.target_modules):
        mismatches.append(
            "target_modules: expected "
            f"{expected_lora_config.target_modules}, got {stored_lora_config.target_modules}"
        )

    if mismatches:
        raise ValueError(
            "Warm-start adapter is incompatible with the requested LoRA configuration. "
            "Please reuse the same LoRA topology when continuing training. "
            f"Details: {'; '.join(mismatches)}"
        )


def load_train_data(train_jsonl: str | Path) -> list[dict[str, Any]]:
    with open(train_jsonl, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError(f"Training jsonl is empty: {train_jsonl}")
    return records


def split_train_validation_data(
    train_records: list[dict[str, Any]],
    validation_records: list[dict[str, Any]] | None,
    validation_split_ratio: float,
    validation_max_samples: int | None,
    validation_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if validation_records:
        return train_records, validation_records, "external_jsonl"

    if validation_split_ratio <= 0:
        return train_records, [], "disabled"

    if len(train_records) < 2:
        return train_records, [], "insufficient_train_samples"

    val_count = max(1, int(round(len(train_records) * validation_split_ratio)))
    val_count = min(val_count, len(train_records) - 1)
    if validation_max_samples is not None:
        val_count = min(val_count, validation_max_samples)

    if val_count <= 0:
        return train_records, [], "disabled"

    rng = random.Random(validation_seed)
    indices = list(range(len(train_records)))
    rng.shuffle(indices)
    validation_index_set = set(indices[:val_count])

    split_train_records = [record for idx, record in enumerate(train_records) if idx not in validation_index_set]
    split_validation_records = [record for idx, record in enumerate(train_records) if idx in validation_index_set]
    if not split_train_records or not split_validation_records:
        return train_records, [], "insufficient_train_samples"
    return split_train_records, split_validation_records, "train_split"


def compute_batch_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    sub_talker_loss_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    typed_model = cast(Any, model)
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    speaker_embedding_batch = typed_model.speaker_encoder(
        ref_mels.to(device=model_device, dtype=model_dtype)
    ).detach()

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = typed_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = typed_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding_batch

    input_embeddings = input_text_embedding + input_codec_embedding

    for group_idx in range(1, 16):
        codec_i_embedding = typed_model.talker.code_predictor.get_input_embeddings()[group_idx - 1](
            codec_ids[:, :, group_idx]
        )
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = typed_model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]

    _, sub_talker_loss = typed_model.talker.forward_sub_talker_finetune(
        talker_codec_ids,
        talker_hidden_states,
    )
    total_loss = outputs.loss + sub_talker_loss_weight * sub_talker_loss
    return total_loss, outputs.loss, sub_talker_loss


def run_validation_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader | None,
    accelerator: Accelerator,
    sub_talker_loss_weight: float,
) -> dict[str, float | int] | None:
    if dataloader is None:
        return None

    was_training = model.training
    model.eval()

    local_total_loss = 0.0
    local_talker_loss = 0.0
    local_sub_talker_loss = 0.0
    local_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            total_loss, talker_loss, sub_talker_loss = compute_batch_loss(model, batch, sub_talker_loss_weight)
            local_total_loss += float(total_loss.detach().item())
            local_talker_loss += float(talker_loss.detach().item())
            local_sub_talker_loss += float(sub_talker_loss.detach().item())
            local_steps += 1

    if was_training:
        model.train()

    total_loss_tensor = torch.tensor(local_total_loss, device=accelerator.device)
    talker_loss_tensor = torch.tensor(local_talker_loss, device=accelerator.device)
    sub_talker_loss_tensor = torch.tensor(local_sub_talker_loss, device=accelerator.device)
    step_count_tensor = torch.tensor(local_steps, device=accelerator.device)

    reduced_total_loss = cast(torch.Tensor, accelerator.reduce(total_loss_tensor, reduction="sum"))
    reduced_talker_loss = cast(torch.Tensor, accelerator.reduce(talker_loss_tensor, reduction="sum"))
    reduced_sub_talker_loss = cast(torch.Tensor, accelerator.reduce(sub_talker_loss_tensor, reduction="sum"))
    reduced_step_count = cast(torch.Tensor, accelerator.reduce(step_count_tensor, reduction="sum"))

    if int(reduced_step_count.item()) == 0:
        return None

    step_count = int(reduced_step_count.item())
    return {
        "avg_loss": float((reduced_total_loss / reduced_step_count).item()),
        "avg_talker_loss": float((reduced_talker_loss / reduced_step_count).item()),
        "avg_sub_talker_loss": float((reduced_sub_talker_loss / reduced_step_count).item()),
        "num_steps": step_count,
    }


def get_current_lr(optimizer: AdamW) -> float:
    return float(optimizer.param_groups[0]["lr"])


def configure_cpu_threading(accelerator: Accelerator, requested_threads: Any) -> int | None:
    if requested_threads in (None, ""):
        if accelerator.device.type == "cpu":
            return None
        requested_threads = min(4, os.cpu_count() or 1)

    thread_count = max(1, int(requested_threads))
    torch.set_num_threads(thread_count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return thread_count


def save_artifacts(
    output_dir: Path,
    unwrapped_model: torch.nn.Module,
    lora_config,
    speaker_embedding: torch.Tensor | dict[str, Any],
    resolved: ResolvedConfig,
    metrics: dict[str, Any],
    is_multi_speaker: bool = False,
) -> None:
    output_dir = ensure_dir(output_dir)
    save_lora_adapter(unwrapped_model, output_dir / "adapter", lora_config)
    if is_multi_speaker and isinstance(speaker_embedding, dict):
        save_multi_speaker_patch(output_dir / "speaker_embedding.safetensors", speaker_embedding)
        save_json(make_multi_speaker_config_patch(speaker_embedding), output_dir / "config_patch.json")
    else:
        embed = cast(torch.Tensor, speaker_embedding)
        save_speaker_patch(output_dir / "speaker_embedding.safetensors", resolved.speaker_id, embed)
        save_json(make_config_patch(resolved.speaker_name, resolved.speaker_id), output_dir / "config_patch.json")
    save_json(dict(resolved), output_dir / "train_args.json")
    save_json(metrics, output_dir / "metrics.json")


def main() -> None:
    args = parse_args()
    resolved = resolve_config(args)
    torch_dtype = parse_torch_dtype(resolved.torch_dtype)

    output_root = ensure_dir(resolved.output_root)
    train_records = load_train_data(resolved.train_jsonl)
    validation_records = load_train_data(resolved.validation_jsonl) if resolved.validation_jsonl else []
    train_data, validation_data, validation_source = split_train_validation_data(
        train_records=train_records,
        validation_records=validation_records,
        validation_split_ratio=resolved.validation_split_ratio,
        validation_max_samples=resolved.validation_max_samples,
        validation_seed=resolved.validation_seed,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=resolved.gradient_accumulation_steps,
        mixed_precision=resolved.mixed_precision,
    )
    configured_cpu_threads = configure_cpu_threading(accelerator, resolved.cpu_threads)
    if configured_cpu_threads is not None:
        accelerator.print(
            f"CPU thread limit enabled for training/preprocessing: {configured_cpu_threads}"
        )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        resolved.base_model,
        dtype=torch_dtype,
        attn_implementation=resolved.attn_implementation,
        local_files_only=resolved.local_files_only,
    )
    config = cast(Any, AutoConfig.from_pretrained(resolved.base_model, local_files_only=resolved.local_files_only))

    ref_audio_paths = collect_unique_ref_audios(train_data)
    is_multi_speaker: bool
    if resolved.multi_speaker_flag is not None:
        is_multi_speaker = bool(resolved.multi_speaker_flag)
    else:
        is_multi_speaker = len(ref_audio_paths) > 1

    if is_multi_speaker and len(ref_audio_paths) > 1:
        accelerator.print(f"Multi-speaker mode: {len(ref_audio_paths)} unique reference audios detected")
        speaker_name_map: dict[str, str] | None = None
        if resolved.speaker_names_override:
            names = cast(list[str], resolved.speaker_names_override)
            if len(names) == len(ref_audio_paths):
                speaker_name_map = dict(zip(ref_audio_paths, names))
            else:
                accelerator.print(
                    f"WARNING: --speaker_names count ({len(names)}) != ref_audio count ({len(ref_audio_paths)}); "
                    "using auto-inferred names"
                )
        speaker_data = extract_multi_speaker_embeddings(
            qwen3tts,
            ref_audio_paths,
            base_speaker_id=resolved.speaker_base_id,
            speaker_name_map=speaker_name_map,
        )
        speaker_embedding: torch.Tensor | dict[str, Any] = speaker_data
        for ref_path, info in speaker_data.items():
            accelerator.print(f"  Speaker '{info['speaker_name']}' (id={info['speaker_id']}) <- {Path(ref_path).name}")
    else:
        speaker_embedding = extract_target_speaker_embedding(qwen3tts, train_data[0]["ref_audio"])
        accelerator.print(f"Single-speaker mode: '{resolved.speaker_name}' (id={resolved.speaker_id})")

    lora_config = build_lora_config(
        r=resolved.lora_r,
        alpha=resolved.lora_alpha,
        dropout=resolved.lora_dropout,
        target_modules=resolved.target_module_pattern,
        bias=resolved.lora_bias,
    )
    trainable_params, total_params, enabled_extra = inject_lora(
        qwen3tts.model,
        lora_config,
        extra_trainable_modules=resolved.extra_trainable_modules,
    )
    if resolved.init_adapter_dir:
        validate_warm_start_adapter_config(resolved.init_adapter_dir, lora_config)
        load_lora_adapter_weights(qwen3tts.model, resolved.init_adapter_dir)
    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters were enabled. Please check target_modules / extra_trainable_modules."
        )

    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    dataloader = DataLoader(
        dataset,
        batch_size=resolved.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    validation_dataloader: DataLoader | None = None
    if validation_data:
        validation_dataset = TTSDataset(validation_data, qwen3tts.processor, config)
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=resolved.batch_size,
            shuffle=False,
            collate_fn=validation_dataset.collate_fn,
        )

    optimizer = AdamW(
        [param for param in qwen3tts.model.parameters() if param.requires_grad],
        lr=resolved.learning_rate,
        weight_decay=resolved.weight_decay,
    )

    num_update_steps_per_epoch = max(math.ceil(len(dataloader) / resolved.gradient_accumulation_steps), 1)
    num_training_steps = max(num_update_steps_per_epoch * resolved.num_epochs, 1)
    num_warmup_steps = min(int(num_training_steps * resolved.warmup_ratio), max(num_training_steps - 1, 0))
    scheduler = get_scheduler(
        name=resolved.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    if validation_dataloader is not None:
        model, optimizer, dataloader, validation_dataloader, scheduler = accelerator.prepare(
            qwen3tts.model,
            optimizer,
            dataloader,
            validation_dataloader,
            scheduler,
        )
    else:
        model, optimizer, dataloader, scheduler = accelerator.prepare(qwen3tts.model, optimizer, dataloader, scheduler)

    metrics: dict[str, Any] = {
        "base_model": resolved.base_model,
        "speaker_name": resolved.speaker_name,
        "speaker_id": resolved.speaker_id,
        "multi_speaker": is_multi_speaker,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "target_scope": resolved.target_scope,
        "target_module_pattern": resolved.target_module_pattern,
        "target_modules": resolved.target_modules,
        "extra_trainable_modules": enabled_extra,
        "init_adapter_dir": resolved.init_adapter_dir,
        "lr_scheduler_type": resolved.lr_scheduler_type,
        "warmup_ratio": resolved.warmup_ratio,
        "num_training_steps": num_training_steps,
        "num_warmup_steps": num_warmup_steps,
        "cpu_threads": configured_cpu_threads,
        "validation_enabled": bool(validation_data),
        "validation_source": validation_source,
        "validation_samples": len(validation_data),
        "early_stopping_patience": resolved.early_stopping_patience,
        "early_stopping_min_delta": resolved.early_stopping_min_delta,
        "step_losses": [],
        "epoch_losses": [],
        "validation_epoch_losses": [],
    }
    if accelerator.is_main_process and validation_data:
        save_json(validation_data, output_root / "validation_records.json")
    save_json(dict(resolved), output_root / "train_args.json")

    accelerator.print(
        f"LoRA training started | trainable params: {trainable_params:,} / {total_params:,} "
        f"({trainable_params / total_params:.4%})"
    )
    accelerator.print(f"Target modules: {resolved.target_modules}")
    accelerator.print(f"Target scope: {resolved.target_scope}")
    accelerator.print(f"Target module pattern: {resolved.target_module_pattern}")
    if resolved.init_adapter_dir:
        accelerator.print(f"Warm-start adapter: {resolved.init_adapter_dir}")
    accelerator.print(
        f"LR scheduler: {resolved.lr_scheduler_type} | warmup_ratio={resolved.warmup_ratio:.4f} "
        f"| warmup_steps={num_warmup_steps} | total_steps={num_training_steps}"
    )
    if enabled_extra:
        accelerator.print(f"Extra trainable parameters enabled: {len(enabled_extra)}")
    if resolved.extra_trainable_modules:
        accelerator.print(f"Phase-2 optional modules requested: {resolved.extra_trainable_modules}")
    else:
        accelerator.print(f"Phase-2 optional modules available but disabled by default: {PHASE2_OPTIONAL_MODULES}")
    if validation_data:
        accelerator.print(
            f"Validation enabled | source={validation_source} | samples={len(validation_data)} | "
            f"early_stopping_patience={resolved.early_stopping_patience}"
        )
    else:
        accelerator.print("Validation disabled; best-checkpoint selection and early stopping will be skipped.")

    model.train()
    best_validation_loss: float | None = None
    best_checkpoint_epoch: int | None = None
    best_checkpoint_dir: str | None = None
    epochs_without_improvement = 0

    for epoch in range(resolved.num_epochs):
        epoch_losses: list[float] = []
        should_stop = False

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(model):
                loss, talker_loss, sub_talker_loss = compute_batch_loss(
                    model,
                    batch,
                    resolved.sub_talker_loss_weight,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        resolved.max_grad_norm,
                    )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            loss_value = float(loss.detach().item())
            epoch_losses.append(loss_value)

            if step % resolved.logging_steps == 0:
                current_lr = get_current_lr(optimizer)
                accelerator.print(
                    f"Epoch {epoch} | Step {step} | Loss: {loss_value:.4f} | "
                    f"Talker: {float(talker_loss.detach().item()):.4f} | "
                    f"SubTalker: {float(sub_talker_loss.detach().item()):.4f} | LR: {current_lr:.8f}"
                )
                metrics["step_losses"].append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": loss_value,
                        "talker_loss": float(talker_loss.detach().item()),
                        "sub_talker_loss": float(sub_talker_loss.detach().item()),
                        "lr": current_lr,
                    }
                )

        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)
        metrics["epoch_losses"].append(
            {
                "epoch": epoch,
                "avg_loss": avg_epoch_loss,
                "num_steps": len(epoch_losses),
                "last_lr": get_current_lr(optimizer),
            }
        )
        accelerator.print(f"Epoch {epoch} completed | avg_loss={avg_epoch_loss:.4f}")

        validation_result = run_validation_epoch(
            model=model,
            dataloader=validation_dataloader,
            accelerator=accelerator,
            sub_talker_loss_weight=resolved.sub_talker_loss_weight,
        )
        if validation_result is not None:
            validation_record = {"epoch": epoch, **validation_result}
            metrics["validation_epoch_losses"].append(validation_record)
            accelerator.print(
                f"Validation epoch {epoch} | avg_loss={validation_result['avg_loss']:.4f} | "
                f"avg_talker_loss={validation_result['avg_talker_loss']:.4f} | "
                f"avg_sub_talker_loss={validation_result['avg_sub_talker_loss']:.4f}"
            )

            improved = best_validation_loss is None or (
                best_validation_loss - float(validation_result["avg_loss"])
            ) > resolved.early_stopping_min_delta
            if improved:
                best_validation_loss = float(validation_result["avg_loss"])
                best_checkpoint_epoch = epoch
                best_checkpoint_dir = str(output_root / "best-checkpoint")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            metrics["best_checkpoint"] = {
                "epoch": best_checkpoint_epoch,
                "avg_validation_loss": best_validation_loss,
                "path": best_checkpoint_dir,
            }

            if (
                resolved.early_stopping_patience > 0
                and epochs_without_improvement >= resolved.early_stopping_patience
            ):
                accelerator.print(
                    f"Early stopping triggered at epoch {epoch}: no validation improvement for "
                    f"{epochs_without_improvement} consecutive validation checks."
                )
                should_stop = True

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_artifacts(output_root / f"checkpoint-epoch-{epoch}", unwrapped_model, lora_config, speaker_embedding, resolved, metrics, is_multi_speaker)
            save_artifacts(output_root, unwrapped_model, lora_config, speaker_embedding, resolved, metrics, is_multi_speaker)
            if best_checkpoint_epoch == epoch and validation_result is not None:
                save_artifacts(output_root / "best-checkpoint", unwrapped_model, lora_config, speaker_embedding, resolved, metrics, is_multi_speaker)

        if should_stop:
            break

    if best_checkpoint_epoch is not None:
        accelerator.print(
            f"LoRA fine-tuning finished. Latest artifacts: {output_root} | "
            f"Best checkpoint: epoch {best_checkpoint_epoch} -> {output_root / 'best-checkpoint'}"
        )
    else:
        accelerator.print(f"LoRA fine-tuning finished. Artifacts saved to: {output_root}")


if __name__ == "__main__":
    main()
