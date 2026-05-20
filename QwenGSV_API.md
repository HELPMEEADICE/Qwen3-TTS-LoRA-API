# Qwen3-TTS-12Hz-1.7B-Base API 文档

基于 FastAPI 的语音克隆 API，接口兼容 GPT-SoVITS 调用方式。

## 目录

- [启动服务](#启动服务)
- [推理 `/`](#推理-)
- [更换默认参考音频 `/change_refer`](#更换默认参考音频-changerere)
- [命令控制 `/control`](#命令控制-control)
- [参数说明](#参数说明)
- [调用示例](#调用示例)

---

## 启动服务

```bash
cd G:\yzylauncher-qwen3tts-win\python
python api-qwen.py -m "./Qwen3-TTS-12Hz-1.7B-Base" -a 0.0.0.0 -p 9880
```

参考音频在每次请求中传入，启动时无需指定。

### 启动参数

| 参数 | 必填 | 默认值 | 说明 |
|------|:--:|--------|------|
| `-m` | 否 | `./Qwen3-TTS-12Hz-1.7B-Base` | 模型路径 |
| `-dr` | 否 | `""` | 默认参考音频路径（可选，设置后请求可不传参考音频） |
| `-dt` | 否 | `""` | 默认参考音频文本（可选，需配合 `-dr` 使用） |
| `-d` | 否 | `cuda:0` | 推理设备，`cuda:0` / `cpu` |
| `-a` | 否 | `0.0.0.0` | 绑定地址 |
| `-p` | 否 | `9880` | 绑定端口 |
| `-fp` | 否 | - | 使用 float32 全精度（默认 bfloat16） |
| `-hp` | 否 | - | 使用 float16 半精度（默认 bfloat16） |
| `--flash-attn` | 否 | - | 启用 FlashAttention 2（默认使用 SDPA） |
| `-sm` | 否 | `close` | 流式返回模式: `close`(关闭) / `normal`(分段逐段返回) |
| `-mt` | 否 | 流式默认`ogg`, 非流式`wav` | 音频编码格式: `wav` / `ogg` |
| `-cp` | 否 | `""` | 默认文本切分符号，如 `",.。!！?？"` |

---

## 推理 `/`

语音克隆推理，返回 WAV 音频。**支持两种模式：**

| 模式 | `prompt_text` | 说明 |
|------|:--:|------|
| **ICL**（上下文学习） | 提供 | 利用参考文本+音频获取更好的音色和韵律克隆效果 |
| **x_vector_only** | 不提供 | 仅用参考音频提取音色嵌入，**不需要参考文本** |

### 请求

**GET**

```
http://127.0.0.1:9880?refer_wav_path=<参考音频>&text=<合成文本>&text_language=<语言>[&prompt_text=<参考文本>][&参数...]
```

**POST（含参考文本，ICL 模式）**

```json
{
    "refer_wav_path": "参考音频.wav",
    "prompt_text": "参考音频的文本。",
    "text": "要合成的文本",
    "text_language": "Chinese",
    "cut_punc": ",.。!！?？",
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9
}
```

**POST（不含参考文本，x_vector_only 模式）**

```json
{
    "refer_wav_path": "参考音频.wav",
    "text": "要合成的文本",
    "text_language": "Chinese"
}
```

### 请求参数

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|:--:|------|--------|------|
| `text` | 是 | string | - | 要合成的文本 |
| `text_language` | 是 | string | - | 合成文本语言，见[语言代码](#语言代码) |
| `refer_wav_path` | 是* | string | - | 参考音频路径（URL/本地路径/base64）。若启动时设置了 `-dr` 则可不传 |
| `prompt_text` | 否 | string | - | 参考音频对应文本。提供则走 ICL 模式（效果更好），不提供则走 x_vector_only 模式 |
| `stream_mode` | 否 | string | `close` | 流式模式: `close` 一次性返回 / `normal` 分段逐段返回（需配合 `cut_punc` 使用） |
| `media_type` | 否 | string | 流式默认`ogg` | 音频编码: `wav` / `ogg`（流式推荐 `ogg`） |
| `cut_punc` | 否 | string | `-cp` 参数值 | 文本切分符号，用这些符号将文本分段后逐段合成 |
| `top_k` | 否 | int | 50 | Top-K 采样参数 |
| `top_p` | 否 | float | 1.0 | Top-P (nucleus) 采样参数 |
| `temperature` | 否 | float | 0.9 | 采样温度，值越高随机性越大 |
| `repetition_penalty` | 否 | float | 1.05 | 重复惩罚系数 |
| `max_new_tokens` | 否 | int | 2048 | 最大生成 token 数 |

> \* 未设置 `-dr` 时必填。

### 响应

- **成功 (200)**: 直接返回 WAV 音频二进制流，`Content-Type: audio/wav`
- **失败 (400)**: JSON 格式错误信息

```json
{
    "code": 400,
    "message": "错误描述"
}
```

### 场景示例

**1) 不提供参考文本（x_vector_only 模式）**

```bash
curl "http://127.0.0.1:9880?refer_wav_path=./test.wav&text=你好世界&text_language=Chinese" -o output.wav
```

**2) 提供参考文本（ICL 模式，效果更好）**

```bash
curl "http://127.0.0.1:9880?refer_wav_path=./test.wav&prompt_text=一二三四五。&text=你好世界&text_language=Chinese" -o output.wav
```

**3) 如果启动时设置了 `-dr`/`-dt`，请求可省略参考音频**

```bash
curl "http://127.0.0.1:9880?text=你好世界&text_language=Chinese" -o output.wav
```

**4) POST 方式 + 自定义生成参数**

```bash
curl -X POST "http://127.0.0.1:9880/" \
  -H "Content-Type: application/json" \
  -d '{
    "refer_wav_path": "./test.wav",
    "prompt_text": "一二三四五。",
    "text": "你好，这是一个测试。",
    "text_language": "Chinese",
    "top_k": 30,
    "top_p": 0.95,
    "temperature": 0.8
  }' -o output.wav
```

**5) 带文本切分（长文本自动分段合成）**

```bash
curl "http://127.0.0.1:9880?refer_wav_path=./test.wav&prompt_text=一二三。&text=第一句。第二句！第三句？&text_language=Chinese&cut_punc=。！？" -o output.wav
```

> `cut_punc` 支持符号: `，` `。` `？` `！` `,` `.` `?` `!` `~` `:` `：` `—` `…`

**6) 流式返回（分段逐段输出，适合长文本）**

```bash
# 非流式（默认）: 所有分段合成完后一次性返回
curl "http://127.0.0.1:9880?refer_wav_path=./test.wav&prompt_text=参考。&text=第一句。第二句。第三句。&text_language=Chinese&cut_punc=。" -o output.wav

# 流式: 每个分段合成完立即返回，客户端可边收边播
curl "http://127.0.0.1:9880?refer_wav_path=./test.wav&prompt_text=参考。&text=第一句。第二句。第三句。&text_language=Chinese&stream_mode=normal&cut_punc=。" -o output.wav
```

```python
# Python 流式接收
import requests

r = requests.post("http://127.0.0.1:9880/", json={
    "refer_wav_path": "./test.wav",
    "prompt_text": "参考文本。",
    "text": "第一句。第二句。第三句。",
    "text_language": "Chinese",
    "stream_mode": "normal",
    "cut_punc": "。",
}, stream=True)

for chunk in r.iter_content(chunk_size=None):
    # chunk 为逐段音频，可立即播放
    with open("stream_output.wav", "ab") as f:
        f.write(chunk)
```

---

## 更换默认参考音频 `/change_refer`

动态更换默认参考音频。`prompt_text` 可选（不传则后续走 x_vector_only 模式）。

### 请求

**GET**

```
http://127.0.0.1:9880/change_refer?refer_wav_path=新参考.wav
```

**POST**

```json
{
    "refer_wav_path": "新参考.wav",
    "prompt_text": "新参考文本。"
}
```

| 参数 | 必填 | 类型 | 说明 |
|------|:--:|------|------|
| `refer_wav_path` | 是 | string | 参考音频路径 |
| `prompt_text` | 否 | string | 参考音频对应文本 |

### 响应

```json
{"code": 0, "message": "Success"}
```

---

## 命令控制 `/control`

控制 API 服务进程。

### 请求

**GET**

```
http://127.0.0.1:9880/control?command=restart
```

**POST**

```json
{"command": "exit"}
```

| 参数 | 必填 | 类型 | 说明 |
|------|:--:|------|------|
| `command` | 是 | string | `"restart"` 重新启动进程 / `"exit"` 结束进程 |

### 响应

无响应体，进程直接退出或重启。

---

## 语言代码

| 代码 | 语言 |
|------|------|
| `Chinese` / `zh` / `中文` | 中文 |
| `English` / `en` / `英文` | 英文 |
| `Japanese` / `ja` / `日文` | 日文 |
| `Korean` / `ko` / `韩文` | 韩文 |
| `German` / `de` / `德文` | 德文 |
| `French` / `fr` / `法文` | 法文 |
| `Russian` / `ru` / `俄文` | 俄文 |
| `Portuguese` / `pt` / `葡萄牙语` | 葡萄牙语 |
| `Spanish` / `es` / `西班牙语` | 西班牙语 |
| `Italian` / `it` / `意大利语` | 意大利语 |
| `Auto` / `auto` / `多语种混合` | 自动检测 |

> `text_language` 同时支持 SoVITS 风格代码和 Qwen3-TTS 完整名称。

---

## 与 GPT-SoVITS API 的差异

| 项目 | GPT-SoVITS | Qwen3-TTS |
|------|------------|-----------|
| 参考文本 | 必需 | **可选**（不提供时走 x_vector_only 模式） |
| 音频编码 | WAV / OGG / AAC | WAV / OGG |
| 流式返回 | 支持 (normal/keepalive) | **支持伪流式** (按切分符号分段逐段返回) |
| `prompt_language` | 需要 | 不需要（从文本自动推断） |
| `speed` | 支持 | 不支持 |
| `inp_refs` | 支持多参考融合 | 暂不支持 |
| `sample_steps` | 支持 | 不支持 |
| `if_sr` | 支持超分辨率 | 不支持 |
| 返回数据类型 | int16 / int32 | float32 → int16（自动转换） |

---

## 完整调用示例

### curl

```bash
# 不提供参考文本（x_vector_only 模式）
curl "http://127.0.0.1:9880?refer_wav_path=./sample.wav&text=你好，这是合成结果。&text_language=Chinese" -o output.wav

# 提供参考文本（ICL 模式，效果更好）
curl "http://127.0.0.1:9880?refer_wav_path=./sample.wav&prompt_text=这是参考语音的文本。&text=你好，这是合成结果。&text_language=Chinese" -o output.wav
```

### Python

```python
import requests

# x_vector_only 模式（仅音色克隆，不需要参考文本）
r = requests.post("http://127.0.0.1:9880/", json={
    "refer_wav_path": "./my_voice.wav",
    "text": "用我的声音说出这句话。",
    "text_language": "Chinese",
})
with open("output.wav", "wb") as f:
    f.write(r.content)

# ICL 模式（提供参考文本，效果更好）
r = requests.post("http://127.0.0.1:9880/", json={
    "refer_wav_path": "./my_voice.wav",
    "prompt_text": "这是我的声音样本。",
    "text": "用我的声音说出这句话。",
    "text_language": "Chinese",
    "top_k": 50,
    "top_p": 1.0,
    "temperature": 0.9,
})
with open("output.wav", "wb") as f:
    f.write(r.content)
```

### JavaScript (fetch)

```javascript
const params = new URLSearchParams({
    refer_wav_path: "./sample.wav",
    text: "合成的目标文本",
    text_language: "Chinese",
});
const resp = await fetch(`http://127.0.0.1:9880/?${params}`);
const blob = await resp.blob();
// blob 即为 WAV 音频
```
