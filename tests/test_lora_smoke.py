from __future__ import annotations

import re
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from lora_finetuning import (common, export_custom_voice,
                             generate_yachiyo_multilingual_samples,
                             inference_with_lora, sft_12hz_lora)


class DummyLoRAModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)
        self.k_proj = nn.Linear(8, 8, bias=False)
        self.v_proj = nn.Linear(8, 8, bias=False)
        self.o_proj = nn.Linear(8, 8, bias=False)
        self.gate_proj = nn.Linear(8, 8, bias=False)
        self.up_proj = nn.Linear(8, 8, bias=False)
        self.down_proj = nn.Linear(8, 8, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.q_proj(inputs)
        hidden = self.k_proj(hidden)
        hidden = self.v_proj(hidden)
        hidden = self.o_proj(hidden)
        return hidden


class FakeInferenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            tts_model_type="base",
            talker_config=SimpleNamespace(spk_id={}, spk_is_dialect={}),
        )
        self.tts_model_type = "base"
        self.talker = SimpleNamespace(
            model=SimpleNamespace(
                codec_embedding=SimpleNamespace(weight=torch.zeros(4001, 8, dtype=torch.float32))
            )
        )
        self.supported_speakers: list[str] = []
        self.eval_called = False

    def eval(self) -> "FakeInferenceModel":
        self.eval_called = True
        return self


class FakeQwen3TTS:
    def __init__(self) -> None:
        self.model = FakeInferenceModel()
        self.generate_calls: list[dict[str, object]] = []

    def generate_custom_voice(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [np.zeros(240, dtype=np.float32)], 24000


class LoRASmokeTests(unittest.TestCase):
    def make_temp_dir(self) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        return Path(temp_dir.name)

    def build_dummy_lora_model(self) -> tuple[nn.Module, object]:
        model = DummyLoRAModule()
        lora_config = common.build_lora_config(
            r=4,
            alpha=8,
            dropout=0.0,
            target_modules=common.DEFAULT_TARGET_MODULES,
        )
        trainable_params, total_params, enabled_extra = common.inject_lora(model, lora_config)
        self.assertGreater(trainable_params, 0)
        self.assertGreater(total_params, trainable_params)
        self.assertEqual(enabled_extra, [])
        return model, lora_config

    def test_resolve_config_prefers_cli_over_yaml(self) -> None:
        work_dir = self.make_temp_dir()
        config_path = work_dir / "smoke_config.yaml"
        config_path.write_text(
                        "\n".join(
                                [
                                        "model:",
                                        "  base_model: models/from-yaml",
                                        "training:",
                                        "  batch_size: 2",
                                        "lora:",
                                        "  target_modules:",
                                        "    - q_proj",
                                        "    - o_proj",
                                        "data:",
                                        "  speaker_name: yaml_speaker",
                                        "artifacts:",
                                        "  output_root: outputs/from-yaml",
                                        "  init_adapter_dir: outputs/init-adapter",
                                ]
                        ),
            encoding="utf-8",
        )

        args = Namespace(
            config=str(config_path),
            base_model=None,
            train_jsonl=None,
            output_root=None,
            init_adapter_dir=None,
            speaker_name="cli_speaker",
            speaker_id=3001,
            torch_dtype="fp16",
            attn_implementation=None,
            local_files_only=True,
            batch_size=1,
            gradient_accumulation_steps=None,
            learning_rate=None,
            weight_decay=None,
            num_epochs=None,
            warmup_ratio=None,
            lr_scheduler_type=None,
            max_grad_norm=None,
            mixed_precision=None,
            logging_steps=None,
            sub_talker_loss_weight=None,
            validation_jsonl=None,
            validation_split_ratio=None,
            validation_max_samples=None,
            validation_seed=None,
            early_stopping_patience=None,
            early_stopping_min_delta=None,
            lora_r=None,
            lora_alpha=None,
            lora_dropout=None,
            lora_bias=None,
            target_scope=None,
            target_module_regex=None,
            target_modules=["q_proj", "v_proj"],
            extra_trainable_modules=["linear_fc1"],
        )

        resolved = sft_12hz_lora.resolve_config(args)

        self.assertEqual(resolved.base_model, "models/from-yaml")
        self.assertEqual(resolved.batch_size, 1)
        self.assertEqual(resolved.speaker_name, "cli_speaker")
        self.assertEqual(resolved.speaker_id, 3001)
        self.assertEqual(resolved.torch_dtype, "fp16")
        self.assertEqual(resolved.target_modules, ["q_proj", "v_proj"])
        self.assertEqual(resolved.target_scope, common.DEFAULT_TARGET_SCOPE)
        self.assertRegex(resolved.target_module_pattern, r"talker\\.model\\.layers")
        self.assertEqual(resolved.extra_trainable_modules, ["linear_fc1"])
        self.assertEqual(resolved.init_adapter_dir, "outputs/init-adapter")
        self.assertTrue(resolved.local_files_only)

    def test_resolve_config_prefers_cli_init_adapter_dir_over_yaml(self) -> None:
        work_dir = self.make_temp_dir()
        config_path = work_dir / "warm_start_config.yaml"
        config_path.write_text(
            """
artifacts:
  init_adapter_dir: outputs/from-yaml
            """.strip(),
            encoding="utf-8",
        )

        args = Namespace(
            config=str(config_path),
            base_model=None,
            train_jsonl=None,
            output_root=None,
            init_adapter_dir="outputs/from-cli",
            speaker_name=None,
            speaker_id=None,
            torch_dtype=None,
            attn_implementation=None,
            local_files_only=False,
            batch_size=None,
            gradient_accumulation_steps=None,
            learning_rate=None,
            weight_decay=None,
            num_epochs=None,
            warmup_ratio=None,
            lr_scheduler_type=None,
            max_grad_norm=None,
            mixed_precision=None,
            logging_steps=None,
            sub_talker_loss_weight=None,
            validation_jsonl=None,
            validation_split_ratio=None,
            validation_max_samples=None,
            validation_seed=None,
            early_stopping_patience=None,
            early_stopping_min_delta=None,
            lora_r=None,
            lora_alpha=None,
            lora_dropout=None,
            lora_bias=None,
            target_scope=None,
            target_module_regex=None,
            target_modules=None,
            extra_trainable_modules=None,
        )

        resolved = sft_12hz_lora.resolve_config(args)

        self.assertEqual(resolved.init_adapter_dir, "outputs/from-cli")

    def test_build_target_module_regex_defaults_to_talker_only(self) -> None:
        pattern = common.build_target_module_regex(["q_proj", "o_proj"], target_scope="talker_only")

        self.assertIsNotNone(re.match(pattern, "talker.model.layers.0.self_attn.q_proj"))
        self.assertIsNotNone(re.match(pattern, "talker.model.layers.12.self_attn.o_proj"))
        self.assertIsNone(re.match(pattern, "talker.code_predictor.model.layers.0.self_attn.q_proj"))

    def test_split_train_validation_data_is_deterministic(self) -> None:
        records = [{"id": idx} for idx in range(10)]

        train_records, validation_records, source = sft_12hz_lora.split_train_validation_data(
            train_records=records,
            validation_records=None,
            validation_split_ratio=0.2,
            validation_max_samples=3,
            validation_seed=123,
        )
        train_records_2, validation_records_2, source_2 = sft_12hz_lora.split_train_validation_data(
            train_records=records,
            validation_records=None,
            validation_split_ratio=0.2,
            validation_max_samples=3,
            validation_seed=123,
        )

        self.assertEqual(source, "train_split")
        self.assertEqual(source_2, "train_split")
        self.assertEqual(len(train_records), 8)
        self.assertEqual(len(validation_records), 2)
        self.assertEqual([record["id"] for record in train_records], [record["id"] for record in train_records_2])
        self.assertEqual(
            [record["id"] for record in validation_records],
            [record["id"] for record in validation_records_2],
        )

    def test_chunked_batches_jobs_for_generation(self) -> None:
        items = [{"id": idx} for idx in range(7)]

        batches = generate_yachiyo_multilingual_samples.chunked(items, batch_size=3)

        self.assertEqual([len(batch) for batch in batches], [3, 3, 1])
        self.assertEqual(batches[0][0]["id"], 0)
        self.assertEqual(batches[-1][0]["id"], 6)

    def test_build_generation_jobs_preserves_preset_order(self) -> None:
        output_dir = self.make_temp_dir() / "samples"
        presets = [
            {
                "speaker_id": "preset_a",
                "style": "A",
                "description": "desc a",
                "style_instruct": "ins a",
                "prompt_texts": {
                    "Japanese": "こんにちは",
                    "English": "hello",
                },
            },
            {
                "speaker_id": "preset_b",
                "style": "B",
                "description": "desc b",
                "style_instruct": "ins b",
                "prompt_texts": {
                    "Chinese": "你好",
                },
            },
        ]

        jobs, preset_reports = generate_yachiyo_multilingual_samples.build_generation_jobs(
            presets=presets,
            output_dir=output_dir,
            speaker_name="yachiyo_formal",
        )

        self.assertEqual(len(jobs), 3)
        self.assertEqual([job["preset_id"] for job in jobs], ["preset_a", "preset_a", "preset_b"])
        self.assertEqual(jobs[0]["speaker_name"], "yachiyo_formal")
        self.assertTrue(str(jobs[0]["output_path"]).endswith("preset_a\\ja.wav") or str(jobs[0]["output_path"]).endswith("preset_a/ja.wav"))
        self.assertEqual(list(preset_reports.keys()), ["preset_a", "preset_b"])
        self.assertEqual(preset_reports["preset_a"]["style_instruct"], "ins a")

    def test_save_artifacts_writes_expected_training_outputs(self) -> None:
        output_dir = self.make_temp_dir() / "artifacts"
        model, lora_config = self.build_dummy_lora_model()
        resolved = sft_12hz_lora.ResolvedConfig(
            base_model="models/Qwen3-TTS-12Hz-1.7B-Base",
            train_jsonl="resources/segments/train_with_codes_qwen3tts.jsonl",
            output_root=str(output_dir),
            speaker_name="smoke_speaker",
            speaker_id=3000,
        )
        metrics = {"epoch_losses": [{"epoch": 0, "avg_loss": 0.1234, "num_steps": 1}]}
        speaker_embedding = torch.randn(8)

        sft_12hz_lora.save_artifacts(output_dir, model, lora_config, speaker_embedding, resolved, metrics)

        self.assertTrue((output_dir / "adapter" / "adapter_model.safetensors").exists())
        self.assertTrue((output_dir / "adapter" / "adapter_config.json").exists())
        self.assertTrue((output_dir / "config_patch.json").exists())
        self.assertTrue((output_dir / "train_args.json").exists())
        self.assertTrue((output_dir / "metrics.json").exists())
        self.assertTrue((output_dir / "speaker_embedding.safetensors").exists())

        speaker_id, loaded_embedding = common.load_speaker_patch(output_dir / "speaker_embedding.safetensors")
        self.assertEqual(speaker_id, 3000)
        self.assertEqual(tuple(loaded_embedding.shape), (8,))

        config_patch = common.load_json(output_dir / "config_patch.json")
        self.assertEqual(config_patch["tts_model_type"], "custom_voice")
        self.assertEqual(config_patch["talker_config"]["spk_id"]["smoke_speaker"], 3000)

    def test_save_artifacts_multi_speaker_writes_combined_patch(self) -> None:
        output_dir = self.make_temp_dir() / "artifacts_multi"
        model, lora_config = self.build_dummy_lora_model()
        resolved = sft_12hz_lora.ResolvedConfig(
            base_model="models/Qwen3-TTS-12Hz-1.7B-Base",
            train_jsonl="resources/segments/train_with_codes_qwen3tts.jsonl",
            output_root=str(output_dir),
            speaker_name="speaker_a",
            speaker_id=3000,
            speaker_base_id=3000,
            speaker_names_override=None,
            multi_speaker_flag=True,
        )
        metrics = {"epoch_losses": []}
        multi_embedding = {
            "ref_a.wav": {"speaker_name": "alice", "speaker_id": 3000, "embedding": torch.randn(8)},
            "ref_b.wav": {"speaker_name": "bob", "speaker_id": 3001, "embedding": torch.randn(8)},
        }

        sft_12hz_lora.save_artifacts(
            output_dir, model, lora_config, multi_embedding, resolved, metrics, is_multi_speaker=True,
        )

        self.assertTrue((output_dir / "speaker_embedding.safetensors").exists())
        config_patch = common.load_json(output_dir / "config_patch.json")
        self.assertEqual(config_patch["talker_config"]["spk_id"]["alice"], 3000)
        self.assertEqual(config_patch["talker_config"]["spk_id"]["bob"], 3001)

        patches = common.load_multi_speaker_patch(output_dir / "speaker_embedding.safetensors")
        self.assertIn(3000, patches)
        self.assertIn(3001, patches)
        self.assertEqual(tuple(patches[3000].shape), (8,))
        self.assertEqual(tuple(patches[3001].shape), (8,))

        speaker_id, embedding = common.load_speaker_patch(output_dir / "speaker_embedding.safetensors")
        self.assertIn(speaker_id, [3000, 3001])
        self.assertEqual(tuple(embedding.shape), (8,))

    def test_multi_speaker_config_patch_aggregates_all_speakers(self) -> None:
        speakers = {
            "ref_a.wav": {"speaker_name": "alice", "speaker_id": 3000, "embedding": torch.randn(8)},
            "ref_b.wav": {"speaker_name": "bob", "speaker_id": 3001, "embedding": torch.randn(8)},
            "ref_c.wav": {"speaker_name": "carol", "speaker_id": 3002, "embedding": torch.randn(8)},
        }
        config_patch = common.make_multi_speaker_config_patch(speakers)
        self.assertEqual(config_patch["tts_model_type"], "custom_voice")
        spk_id = config_patch["talker_config"]["spk_id"]
        self.assertEqual(spk_id["alice"], 3000)
        self.assertEqual(spk_id["bob"], 3001)
        self.assertEqual(spk_id["carol"], 3002)
        self.assertFalse(config_patch["talker_config"]["spk_is_dialect"]["alice"])

    def test_collect_unique_ref_audios_deduplicates_paths(self) -> None:
        train_data = [
            {"ref_audio": "voice/ref_alice.wav", "audio": "a.wav", "text": "hello"},
            {"ref_audio": "voice/ref_bob.wav", "audio": "b.wav", "text": "world"},
            {"ref_audio": "voice/ref_alice.wav", "audio": "c.wav", "text": "test"},
        ]
        refs = common.collect_unique_ref_audios(train_data)
        self.assertEqual(len(refs), 2)
        self.assertIn("voice/ref_alice.wav", refs)
        self.assertIn("voice/ref_bob.wav", refs)

    def test_infer_speaker_name_from_ref_path(self) -> None:
        self.assertEqual(common.infer_speaker_name_from_ref_path("voice/ref_alice.wav"), "alice")
        self.assertEqual(common.infer_speaker_name_from_ref_path("voice/speaker_bob.wav"), "bob")
        self.assertEqual(common.infer_speaker_name_from_ref_path("voice/carol.wav"), "carol")

    def test_export_custom_voice_bundles_artifacts_and_manifest(self) -> None:
        source_dir = self.make_temp_dir() / "train_outputs"
        output_dir = self.make_temp_dir() / "bundle"
        model, lora_config = self.build_dummy_lora_model()
        resolved = sft_12hz_lora.ResolvedConfig(
            base_model="models/Qwen3-TTS-12Hz-1.7B-Base",
            train_jsonl="resources/segments/train_with_codes_qwen3tts.jsonl",
            output_root=str(source_dir),
            speaker_name="bundle_speaker",
            speaker_id=3002,
        )
        sft_12hz_lora.save_artifacts(
            source_dir,
            model,
            lora_config,
            torch.randn(8),
            resolved,
            metrics={"ok": True},
        )

        argv = [
            "export_custom_voice.py",
            "--base_model",
            "models/Qwen3-TTS-12Hz-1.7B-Base",
            "--source_dir",
            str(source_dir),
            "--output_dir",
            str(output_dir),
        ]
        with patch.object(sys, "argv", argv):
            export_custom_voice.main()

        self.assertTrue((output_dir / "adapter" / "adapter_model.safetensors").exists())
        self.assertTrue((output_dir / "config_patch.json").exists())
        self.assertTrue((output_dir / "speaker_embedding.safetensors").exists())
        self.assertTrue((output_dir / "manifest.json").exists())

        manifest = common.load_json(output_dir / "manifest.json")
        self.assertEqual(manifest["base_model_path"], "models/Qwen3-TTS-12Hz-1.7B-Base")
        self.assertEqual(manifest["speaker_name"], "bundle_speaker")
        self.assertEqual(manifest["speaker_id"], 3002)
        self.assertEqual(manifest["adapter_dir"], "adapter")

    def test_warm_start_adapter_weights_can_be_loaded_after_inject(self) -> None:
        source_dir = self.make_temp_dir() / "source_adapter"
        source_model, lora_config = self.build_dummy_lora_model()
        source_param_name, source_param = next(
            (name, param) for name, param in source_model.named_parameters() if "lora_A" in name
        )
        with torch.no_grad():
            source_param.copy_(torch.full_like(source_param, 0.25))
        common.save_lora_adapter(source_model, source_dir, lora_config)

        target_model = DummyLoRAModule()
        common.inject_lora(target_model, lora_config)
        sft_12hz_lora.validate_warm_start_adapter_config(source_dir, lora_config)
        common.load_lora_adapter_weights(target_model, source_dir)

        target_param = dict(target_model.named_parameters())[source_param_name]
        self.assertTrue(torch.allclose(target_param.detach(), source_param.detach()))

    def test_warm_start_config_mismatch_is_rejected(self) -> None:
        source_dir = self.make_temp_dir() / "source_adapter"
        source_model, lora_config = self.build_dummy_lora_model()
        common.save_lora_adapter(source_model, source_dir, lora_config)

        mismatched_lora_config = common.build_lora_config(
            r=8,
            alpha=8,
            dropout=0.0,
            target_modules=common.DEFAULT_TARGET_MODULES,
        )

        with self.assertRaisesRegex(ValueError, "Warm-start adapter is incompatible"):
            sft_12hz_lora.validate_warm_start_adapter_config(source_dir, mismatched_lora_config)

    def test_inference_main_loads_bundle_and_writes_output(self) -> None:
        bundle_dir = self.make_temp_dir() / "bundle"
        adapter_dir = bundle_dir / "adapter"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"stub")
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")

        common.save_json(common.make_config_patch("inference_speaker", 3000), bundle_dir / "config_patch.json")
        common.save_speaker_patch(bundle_dir / "speaker_embedding.safetensors", 3000, torch.randn(8))
        common.save_json(
            common.build_bundle_manifest(
                base_model_path="models/Qwen3-TTS-12Hz-1.7B-Base",
                speaker_name="inference_speaker",
                speaker_id=3000,
            ),
            bundle_dir / "manifest.json",
        )

        fake_qwen3tts = FakeQwen3TTS()
        output_wav = bundle_dir / "outputs" / "smoke.wav"
        argv = [
            "inference_with_lora.py",
            "--bundle_dir",
            str(bundle_dir),
            "--text",
            "这是一次 LoRA smoke test。",
            "--output_wav",
            str(output_wav),
        ]

        with patch("lora_finetuning.inference_with_lora.Qwen3TTSModel.from_pretrained", return_value=fake_qwen3tts) as from_pretrained, \
             patch("lora_finetuning.inference_with_lora.load_lora_adapter") as load_lora_adapter, \
             patch("lora_finetuning.inference_with_lora.sf.write") as sf_write:
            with patch.object(sys, "argv", argv):
                inference_with_lora.main()

        from_pretrained.assert_called_once()
        call_kwargs = from_pretrained.call_args.kwargs
        self.assertEqual(call_kwargs["device_map"], "cuda:0")
        self.assertEqual(call_kwargs["attn_implementation"], "sdpa")
        self.assertEqual(call_kwargs["dtype"], torch.bfloat16)

        load_lora_adapter.assert_called_once_with(fake_qwen3tts.model, bundle_dir / "adapter")
        self.assertTrue(fake_qwen3tts.model.eval_called)
        self.assertEqual(len(fake_qwen3tts.generate_calls), 1)
        self.assertEqual(fake_qwen3tts.generate_calls[0]["speaker"], "inference_speaker")
        self.assertEqual(fake_qwen3tts.generate_calls[0]["text"], "这是一次 LoRA smoke test。")
        self.assertIn("inference_speaker", fake_qwen3tts.model.supported_speakers)
        self.assertEqual(fake_qwen3tts.model.config.tts_model_type, "custom_voice")
        self.assertEqual(fake_qwen3tts.model.talker.model.codec_embedding.weight[3000].shape[0], 8)

        sf_write.assert_called_once()
        self.assertEqual(Path(sf_write.call_args.args[0]), output_wav)
        self.assertEqual(sf_write.call_args.args[2], 24000)

    def test_load_train_data_rejects_empty_jsonl(self) -> None:
        empty_jsonl = self.make_temp_dir() / "empty.jsonl"
        empty_jsonl.write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Training jsonl is empty"):
            sft_12hz_lora.load_train_data(empty_jsonl)


if __name__ == "__main__":
    unittest.main()
