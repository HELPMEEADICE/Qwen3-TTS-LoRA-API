"""
# api-qwen.py usage

` python api-qwen.py -m "./Qwen3-TTS-12Hz-1.7B-Base" -a 0.0.0.0 -p 9880 `

启动后通过请求参数传入参考音频和文本，无需启动时指定。

基于 faster-qwen3-tts (CUDA Graph 加速, ~5-10x speedup)，支持原生流式输出。

## 执行参数:

 `-m` - `Qwen3-TTS模型路径, 默认"./Qwen3-TTS-12Hz-1.7B-Base"`
 `-dr` - `默认参考音频路径（可选，设置后请求可不传参考音频）`
 `-dt` - `默认参考音频文本（可选，需配合 -dr 使用）`
 `-d` - `推理设备, 默认"cuda:0"`
 `-a` - `绑定地址, 默认"0.0.0.0"`
 `-p` - `绑定端口, 默认9880`
 `-fp` - `使用全精度 float32`
 `-hp` - `使用半精度 float16 (默认 bfloat16)`
 `--flash-attn` - `启用 FlashAttention 2（默认使用 SDPA）`
 `-sm` - `流式返回模式, 默认"close", "close"/"c" 关闭, "normal"/"n" 原生产生流式`
 `-mt` - `音频编码格式, 流式默认"ogg", 非流式默认"wav", "wav"/"ogg"`
 `-cs` - `流式块大小 (codec步数), 默认12 (~1秒), 越小延迟越低`
 `-cp` - `默认文本切分符号, 如",.。!！?？" (仅非流式模式)`
 `--lora` - `LoRA适配器目录名 (相对于 ./lora/), 如"multi_5speakers", 默认使用最大epoch`
 `--lora-epoch` - `指定checkpoint epoch编号 (默认自动选取最大epoch)`
 `--speaker` - `指定说话人名称 (多说话人LoRA时使用)`

## 调用:

### 推理

endpoint: `/`

每次请求必须提供 refer_wav_path，prompt_text 可选。
  - 提供 prompt_text: ICL 模式，合成效果更好
  - 不提供 prompt_text: x_vector_only 模式，仅用音色嵌入克隆（不需要参考文本）

非流式（默认）:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&text=合成文本&text_language=Chinese`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "text": "合成文本",
    "text_language": "Chinese"
}
```

流式返回（原生 CUDA Graph 逐块输出，低延迟，适合长文本）:
GET:
    `http://127.0.0.1:9880?refer_wav_path=123.wav&prompt_text=一二三。&text=第一句。第二句。第三句。&text_language=Chinese&stream_mode=normal&chunk_size=8`
POST:
```json
{
    "refer_wav_path": "123.wav",
    "prompt_text": "一二三。",
    "text": "第一句。第二句。第三句。",
    "text_language": "Chinese",
    "stream_mode": "normal",
    "chunk_size": 8
}
```

RESP:
成功: 返回音频流， http code 200
失败: 返回包含错误信息的 json, http code 400


### 更换默认参考音频

endpoint: `/change_refer`

prompt_text 可选（不传则后续走 x_vector_only 模式）。

GET:
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav`
    `http://127.0.0.1:9880/change_refer?refer_wav_path=123.wav&prompt_text=一二三。`
POST:
```json
{
    "refer_wav_path": "123.wav"
}
```

RESP:
成功: json, http code 200
失败: json, 400


### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
    `http://127.0.0.1:9880/control?command=restart`
POST:
```json
{
    "command": "restart"
}
```

RESP: 无

"""

import argparse
import io
import json
import os
import re
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, Request, Query
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

_script_dir = os.path.dirname(os.path.abspath(__file__))
_sox_path = os.path.join(_script_dir, "sox")
os.environ["PATH"] = _sox_path + os.pathsep + os.environ.get("PATH", "")
_LORA_ROOT = os.path.join(_script_dir, "lora")

if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from faster_qwen3_tts import FasterQwen3TTS
from lora_finetuning.common import (load_json, load_lora_adapter,
                                    apply_config_patch, apply_multi_speaker_patches,
                                    apply_single_speaker_from_multi,
                                    apply_speaker_patch)
from safetensors.torch import load_file


# ---- Language mapping (SoVITS style -> Qwen3-TTS) ----
LANGUAGE_MAP = {
    "zh": "Chinese",
    "中文": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "英文": "English",
    "english": "English",
    "ja": "Japanese",
    "日文": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "韩文": "Korean",
    "korean": "Korean",
    "de": "German",
    "德文": "German",
    "german": "German",
    "fr": "French",
    "法文": "French",
    "french": "French",
    "ru": "Russian",
    "俄文": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "葡萄牙语": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "西班牙语": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "意大利语": "Italian",
    "italian": "Italian",
    "auto": "Auto",
    "多语种混合": "Auto",
}

# Text splitting punctuation (same as SoVITS defaults)
SPLITS = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…"}


def map_language(lang: str) -> str:
    """Map SoVITS-style language codes to Qwen3-TTS language names."""
    if lang is None or lang == "":
        return "Auto"
    key = lang.strip().lower()
    return LANGUAGE_MAP.get(key, LANGUAGE_MAP.get(lang, "Auto"))


def cut_text(text: str, punc: str) -> str:
    """Split text on punctuation for multi-sentence synthesis."""
    punc_list = [p for p in punc if p in SPLITS]
    if not punc_list:
        return text
    punds = r"[" + "".join(re.escape(p) for p in punc_list) + r"]"
    text = text.strip("\n")
    items = re.split(f"({punds})", text)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    if len(items) % 2 == 1:
        mergeitems.append(items[-1])
    text = "\n".join(mergeitems)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    return text


def only_punc(text: str) -> bool:
    return not any(t.isalnum() or t.isalpha() for t in text)


# ---- Default reference ----
class DefaultRefer:
    def __init__(self, path, text=None):
        self.path = path
        self.text = text

    def is_ready(self) -> bool:
        return bool(self.path)


# ---- Global state ----
tts_model: FasterQwen3TTS = None
default_refer: DefaultRefer = None
default_cut_punc: str = ""
stream_mode: str = "close"
media_type: str = "wav"
default_chunk_size: int = 12
device: str = "cuda:0"
lora_speaker_name: str = None


def _is_multi_speaker_patch(patch_file: Path) -> bool:
    state = load_file(str(patch_file))
    return any(k.startswith("embedding_") for k in state)


def _resolve_lora_paths(lora_name: str, epoch: int | None) -> tuple[Path, Path, Path, list[str]]:
    lora_dir = Path(_LORA_ROOT) / lora_name
    if not lora_dir.is_dir():
        raise FileNotFoundError(f"LoRA directory not found: {lora_dir}")

    checkpoint_dirs = sorted(
        [d for d in lora_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-epoch-")],
        key=lambda d: int(d.name.rsplit("-", 1)[-1])
    )

    if epoch is not None:
        target = lora_dir / f"checkpoint-epoch-{epoch}"
        if not target.is_dir():
            raise FileNotFoundError(f"Checkpoint epoch {epoch} not found in {lora_dir}")
    elif checkpoint_dirs:
        target = checkpoint_dirs[-1]
    else:
        target = lora_dir

    adapter_dir = target / "adapter"
    if not adapter_dir.is_dir():
        adapter_dir = lora_dir / "adapter"

    config_patch_file = target / "config_patch.json"
    if not config_patch_file.is_file():
        config_patch_file = lora_dir / "config_patch.json"

    speaker_patch_file = target / "speaker_embedding.safetensors"
    if not speaker_patch_file.is_file():
        speaker_patch_file = lora_dir / "speaker_embedding.safetensors"

    config_patch = load_json(config_patch_file)
    available_speakers = list(config_patch.get("talker_config", {}).get("spk_id", {}).keys())

    return adapter_dir, config_patch_file, speaker_patch_file, available_speakers


def load_model(model_path: str, dtype: torch.dtype, attn_implementation: str,
               lora_name: str = None, lora_epoch: int = None, lora_speaker: str = None):
    global tts_model, lora_speaker_name

    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("CUDA graphs require CUDA device")

    from qwen_tts import Qwen3TTSModel
    from faster_qwen3_tts.predictor_graph import PredictorGraph
    from faster_qwen3_tts.talker_graph import TalkerGraph

    print(f"加载基座模型: {model_path}")
    base_model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
    )

    if lora_name:
        adapter_dir, config_patch_file, speaker_patch_file, available_speakers = \
            _resolve_lora_paths(lora_name, lora_epoch)

        print(f"[LoRA] 加载适配器: {adapter_dir}")
        load_lora_adapter(base_model.model, adapter_dir)
        base_model.model.eval()

        if lora_speaker:
            if lora_speaker not in available_speakers:
                raise ValueError(f"说话人 '{lora_speaker}' 未找到. 可用: {available_speakers}")

            config_patch = load_json(config_patch_file)
            apply_config_patch(base_model.model, config_patch)
            print(f"[LoRA] 配置已应用 (tts_model_type={base_model.model.tts_model_type})")

            if _is_multi_speaker_patch(speaker_patch_file):
                apply_single_speaker_from_multi(
                    base_model.model, speaker_patch_file, config_patch, lora_speaker)
                print(f"[LoRA] 已选择说话人: {lora_speaker}")
            else:
                apply_speaker_patch(base_model.model, speaker_patch_file)
                print(f"[LoRA] 已加载说话人嵌入")

            lora_speaker_name = lora_speaker
        else:
            print(f"[LoRA] 仅注入 LoRA 权重 (保持 voice_clone 模式)")
            print(f"[LoRA] 可用说话人: {available_speakers} (使用 --speaker 切换到自定义语音模式)")

        print("[LoRA] CUDA Graph 将包含 LoRA 权重")

    talker = base_model.model.talker
    talker_config = base_model.model.config.talker_config
    predictor = talker.code_predictor
    pred_config = predictor.model.config
    talker_hidden = talker_config.hidden_size

    print("构建 CUDA Graphs...")
    predictor_graph = PredictorGraph(
        predictor, pred_config, talker_hidden,
        device=device, dtype=dtype, do_sample=True, top_k=50, temperature=0.9,
    )
    talker_graph = TalkerGraph(
        talker.model, talker_config,
        device=device, dtype=dtype, max_seq_len=2048,
    )

    tts_model = FasterQwen3TTS(
        base_model=base_model,
        predictor_graph=predictor_graph,
        talker_graph=talker_graph,
        device=device,
        dtype=dtype,
        max_seq_len=2048,
    )
    print("CUDA Graphs 初始化完成")


def do_tts(
    text: str,
    text_language: str,
    refer_wav_path: str = None,
    prompt_text: str = None,
    top_k: int = None,
    top_p: float = None,
    temperature: float = None,
):
    """Generate TTS audio, return (audio_np, sr)."""
    language = map_language(text_language)

    if lora_speaker_name:
        t_start = time.time()
        wavs, sr = tts_model.generate_custom_voice(
            text=text,
            language=language,
            speaker=lora_speaker_name,
        )
        elapsed = time.time() - t_start

        audio = wavs[0]
        max_val = np.abs(audio).max()
        if max_val > 1:
            audio = audio / max_val

        audio_s = len(audio) / sr if sr > 0 else 0
        steps = int(audio_s * 12)
        it_s = steps / elapsed if elapsed > 0 else 0
        rtf = elapsed / audio_s if audio_s > 0 else 0

        sys.stderr.write(
            f"[custom_voice:{lora_speaker_name}] {steps}步 | {it_s:.1f}it/s | "
            f"耗时:{elapsed:.1f}s | RTF:{rtf:.2f} | 音频:{audio_s:.1f}s\n"
        )
        sys.stderr.flush()
        return audio, int(sr)

    ref_path = refer_wav_path
    ref_text = prompt_text
    if not ref_path:
        if default_refer and default_refer.is_ready():
            ref_path = default_refer.path
            ref_text = ref_text or default_refer.text
        else:
            raise ValueError("未指定参考音频")

    if ref_text and ref_text.strip():
        xvec_only = False
    else:
        xvec_only = True
        ref_text = ""

    t_start = time.time()
    wavs, sr = tts_model.generate_voice_clone(
        text=text,
        language=language,
        ref_audio=ref_path,
        ref_text=ref_text,
        xvec_only=xvec_only,
        do_sample=True,
        top_k=top_k if top_k is not None else 50,
        top_p=top_p if top_p is not None else 1.0,
        temperature=temperature if temperature is not None else 0.9,
        max_new_tokens=2048,
    )
    elapsed = time.time() - t_start

    audio = wavs[0]
    max_val = np.abs(audio).max()
    if max_val > 1:
        audio = audio / max_val

    audio_s = len(audio) / sr if sr > 0 else 0
    steps = int(audio_s * 12)
    it_s = steps / elapsed if elapsed > 0 else 0
    rtf = elapsed / audio_s if audio_s > 0 else 0

    sys.stderr.write(
        f"[非流式] {steps}步 | {it_s:.1f}it/s | "
        f"耗时:{elapsed:.1f}s | RTF:{rtf:.2f} | 音频:{audio_s:.1f}s\n"
    )
    sys.stderr.flush()

    return audio, int(sr)


# ---- Handlers (mirroring api-sovits.py structure) ----

def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *sys.argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


def handle_change(path: str, text: str = None):
    if not path:
        return JSONResponse({"code": 400, "message": '缺少参数: "refer_wav_path"'}, status_code=400)
    default_refer.path = path
    default_refer.text = text if text and text.strip() else None
    return JSONResponse({"code": 0, "message": "Success"}, status_code=200)


def _stream_generate(
    text, text_language, refer_wav_path, prompt_text,
    top_k, top_p, temperature, mt, chunk_size,
):
    """Generator: yield audio chunks via native CUDA-graph streaming."""
    language = map_language(text_language)

    if lora_speaker_name:
        t_start = time.time()
        prefill_ms = 0.0
        total_steps = 0
        try:
            for audio_chunk, sr, timing in tts_model.generate_custom_voice_streaming(
                text=text,
                speaker=lora_speaker_name,
                language=language,
                do_sample=True,
                top_k=top_k if top_k is not None else 50,
                top_p=top_p if top_p is not None else 1.0,
                temperature=temperature if temperature is not None else 0.9,
                max_new_tokens=2048,
                chunk_size=chunk_size,
            ):
                max_val = np.abs(audio_chunk).max()
                if max_val > 1:
                    audio_chunk = audio_chunk / max_val
                if mt == "ogg":
                    buf = io.BytesIO()
                    sf.write(buf, audio_chunk, sr, format="OGG", subtype="VORBIS")
                    yield buf.getvalue()
                else:
                    pcm = (audio_chunk * 32767).astype(np.int16).tobytes()
                    yield pcm

                chunk_idx = timing.get("chunk_index", 0)
                chunk_steps = timing.get("chunk_steps", 0)
                decode_ms = timing.get("decode_ms", 0.0)
                total_steps = timing.get("total_steps_so_far", 0)
                is_final = timing.get("is_final", False)
                if chunk_idx == 0:
                    prefill_ms = timing.get("prefill_ms", 0.0)

                it_s = chunk_steps / (decode_ms / 1000) if decode_ms > 0 else 0
                audio_s = total_steps / 12.0

                if is_final:
                    elapsed = time.time() - t_start
                    avg_it_s = total_steps / elapsed if elapsed > 0 else 0
                    sys.stderr.write(
                        f"\r[custom_voice:{lora_speaker_name} 流式] 总步:{total_steps} | 平均:{avg_it_s:.1f}it/s | "
                        f"预填充:{prefill_ms:.0f}ms | 总耗时:{elapsed:.1f}s | 音频:{audio_s:.1f}s\n"
                    )
                else:
                    sys.stderr.write(
                        f"\r[custom_voice:{lora_speaker_name} 流式] 块{chunk_idx+1} | {chunk_steps}步 | {it_s:.1f}it/s | "
                        f"{decode_ms:.0f}ms | 音频:{audio_s:.1f}s  "
                    )
                sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"\n[错误] {e}\n")
            sys.stderr.flush()
            yield json.dumps({"code": 400, "message": str(e)}).encode()
        return

    ref_path = refer_wav_path
    ref_text = prompt_text
    if not ref_path:
        if default_refer and default_refer.is_ready():
            ref_path = default_refer.path
            ref_text = ref_text or default_refer.text
        else:
            yield json.dumps({"code": 400, "message": "未指定参考音频"}).encode()
            return

    if ref_text and ref_text.strip():
        xvec_only = False
    else:
        xvec_only = True
        ref_text = ""

    t_start = time.time()
    prefill_ms = 0.0
    total_steps = 0
    try:
        for audio_chunk, sr, timing in tts_model.generate_voice_clone_streaming(
            text=text,
            language=language,
            ref_audio=ref_path,
            ref_text=ref_text,
            xvec_only=xvec_only,
            do_sample=True,
            top_k=top_k if top_k is not None else 50,
            top_p=top_p if top_p is not None else 1.0,
            temperature=temperature if temperature is not None else 0.9,
            max_new_tokens=2048,
            chunk_size=chunk_size,
        ):
            max_val = np.abs(audio_chunk).max()
            if max_val > 1:
                audio_chunk = audio_chunk / max_val
            if mt == "ogg":
                buf = io.BytesIO()
                sf.write(buf, audio_chunk, sr, format="OGG", subtype="VORBIS")
                yield buf.getvalue()
            else:
                pcm = (audio_chunk * 32767).astype(np.int16).tobytes()
                yield pcm

            # Progress display
            chunk_idx = timing.get("chunk_index", 0)
            chunk_steps = timing.get("chunk_steps", 0)
            decode_ms = timing.get("decode_ms", 0.0)
            total_steps = timing.get("total_steps_so_far", 0)
            is_final = timing.get("is_final", False)
            if chunk_idx == 0:
                prefill_ms = timing.get("prefill_ms", 0.0)

            it_s = chunk_steps / (decode_ms / 1000) if decode_ms > 0 else 0
            audio_s = total_steps / 12.0

            if is_final:
                elapsed = time.time() - t_start
                avg_it_s = total_steps / elapsed if elapsed > 0 else 0
                sys.stderr.write(
                    f"\r[完成] 总步:{total_steps} | 平均:{avg_it_s:.1f}it/s | "
                    f"预填充:{prefill_ms:.0f}ms | 总耗时:{elapsed:.1f}s | 音频:{audio_s:.1f}s\n"
                )
            else:
                sys.stderr.write(
                    f"\r[流式] 块{chunk_idx+1} | {chunk_steps}步 | {it_s:.1f}it/s | "
                    f"{decode_ms:.0f}ms | 音频:{audio_s:.1f}s  "
                )
            sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"\n[错误] {e}\n")
        sys.stderr.flush()
        yield json.dumps({"code": 400, "message": str(e)}).encode()


def handle(
    refer_wav_path: str,
    prompt_text: str,
    text: str,
    text_language: str,
    cut_punc: str,
    top_k: int,
    top_p: float,
    temperature: float,
    sm: str = None,
    mt: str = None,
    chunk_size: int = None,
):
    if not text:
        return JSONResponse({"code": 400, "message": "缺少text参数"}, status_code=400)

    _sm = sm if sm and sm.lower() in ("normal", "n") else stream_mode
    _mt = mt if mt and mt.lower() in ("wav", "ogg") else media_type
    _cs = chunk_size if chunk_size is not None else default_chunk_size

    if _sm in ("normal", "n"):
        # Native CUDA-graph streaming: full text, progressive audio output
        return StreamingResponse(
            _stream_generate(
                text, text_language, refer_wav_path, prompt_text,
                top_k, top_p, temperature, _mt, _cs,
            ),
            media_type="audio/ogg" if _mt == "ogg" else "audio/wav",
        )

    # Non-streaming: split text on punctuation, generate each segment
    punc = cut_punc if cut_punc is not None else default_cut_punc
    texts = [t.strip() for t in cut_text(text, punc).split("\n") if t.strip() and not only_punc(t)]
    if not texts:
        texts = [text]

    all_audio = []
    common_sr = None
    for t in texts:
        try:
            audio, sr = do_tts(
                text=t,
                text_language=text_language,
                refer_wav_path=refer_wav_path,
                prompt_text=prompt_text,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            all_audio.append(audio)
            if common_sr is None:
                common_sr = sr
        except Exception as e:
            return JSONResponse({"code": 400, "message": str(e)}, status_code=400)

    combined = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, combined, common_sr, format="WAV")
    return StreamingResponse(
        iter([wav_bytes.getvalue()]),
        media_type="audio/wav",
    )


# ---- FastAPI app ----
app = FastAPI()


@app.post("/")
async def tts_endpoint_post(request: Request):
    json_post_raw = await request.json()
    return handle(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
        json_post_raw.get("text"),
        json_post_raw.get("text_language"),
        json_post_raw.get("cut_punc"),
        json_post_raw.get("top_k", None),
        json_post_raw.get("top_p", None),
        json_post_raw.get("temperature", None),
        json_post_raw.get("stream_mode"),
        json_post_raw.get("media_type"),
        json_post_raw.get("chunk_size", default_chunk_size),
    )


@app.get("/")
async def tts_endpoint_get(
    refer_wav_path: str = None,
    prompt_text: str = None,
    text: str = None,
    text_language: str = None,
    cut_punc: str = None,
    top_k: int = None,
    top_p: float = None,
    temperature: float = None,
    stream_mode: str = None,
    media_type: str = None,
    chunk_size: int = None,
):
    return handle(
        refer_wav_path,
        prompt_text,
        text,
        text_language,
        cut_punc,
        top_k,
        top_p,
        temperature,
        stream_mode,
        media_type,
        chunk_size,
    )


@app.post("/change_refer")
async def change_refer_post(request: Request):
    json_post_raw = await request.json()
    return handle_change(
        json_post_raw.get("refer_wav_path"),
        json_post_raw.get("prompt_text"),
    )


@app.get("/change_refer")
async def change_refer_get(
    refer_wav_path: str = None,
    prompt_text: str = None,
):
    return handle_change(refer_wav_path, prompt_text)


@app.post("/control")
async def control_post(request: Request):
    json_post_raw = await request.json()
    return handle_control(json_post_raw.get("command"))


@app.get("/control")
async def control_get(command: str = None):
    return handle_control(command)


# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3-TTS-12Hz-1.7B-Base API")
    parser.add_argument("-m", "--model_path", type=str, default="./Qwen3-TTS-12Hz-1.7B-Base", help="Qwen3-TTS模型路径")
    parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="默认参考音频路径")
    parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="默认参考音频文本")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="推理设备 cuda:0 / cpu")
    parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="绑定地址")
    parser.add_argument("-p", "--port", type=int, default=9880, help="绑定端口")
    parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="使用全精度 float32")
    parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="使用半精度 float16")
    parser.add_argument("--flash-attn", dest="flash_attn", action="store_true", default=False, help="启用 FlashAttention 2 (默认SDPA)")
    parser.add_argument("-cp", "--cut_punc", type=str, default="", help="文本切分符号, 如\",.。!！?？\"")
    parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="流式返回模式: close/c, normal/n")
    parser.add_argument("-mt", "--media_type", type=str, default="wav", help="音频编码格式: wav, ogg")
    parser.add_argument("-cs", "--chunk_size", type=int, default=12, help="流式模式下每块codec步数 (12≈1秒)")
    parser.add_argument("--lora", type=str, default=None, help="LoRA适配器目录名 (相对于 ./lora/), 如 multi_5speakers")
    parser.add_argument("--lora-epoch", type=int, default=None, help="LoRA checkpoint epoch编号 (默认最大)")
    parser.add_argument("--speaker", type=str, default=None, help="多说话人LoRA中指定说话人名称")
    args = parser.parse_args()

    device = args.device
    default_cut_punc = args.cut_punc
    default_chunk_size = args.chunk_size

    # Stream mode
    if args.stream_mode.lower() in ("normal", "n"):
        stream_mode = "normal"
    else:
        stream_mode = "close"

    # Media type
    _mt = args.media_type.lower()
    if stream_mode == "normal":
        media_type = _mt if _mt in ("wav", "ogg") else "ogg"
    else:
        media_type = _mt if _mt in ("wav",) else "wav"

    # Determine dtype
    if args.full_precision and args.half_precision:
        dtype = torch.bfloat16
    elif args.full_precision:
        dtype = torch.float32
    elif args.half_precision:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    attn_impl = "flash_attention_2" if args.flash_attn else "sdpa"

    print(f"设备: {device}")
    print(f"精度: {dtype}")
    print(f"注意力: {attn_impl}")
    print(f"流式模式: {stream_mode}")
    print(f"编码格式: {media_type}")
    print(f"流式块大小: {default_chunk_size} steps (~{default_chunk_size/12:.1f}s)")

    # Set up default reference
    default_refer = DefaultRefer(args.default_refer_path, args.default_refer_text)
    if default_refer.is_ready():
        print(f"默认参考音频路径: {default_refer.path}")
        print(f"默认参考音频文本: {default_refer.text}")
    else:
        print("未指定默认参考音频")

    # Load model
    print(f"加载模型: {args.model_path}")
    load_model(args.model_path, dtype, attn_impl,
               lora_name=args.lora, lora_epoch=getattr(args, 'lora_epoch', None),
               lora_speaker=args.speaker)
    print("模型加载完成")

    uvicorn.run(app, host=args.bind_addr, port=args.port, workers=1)
