# coding=utf-8
"""
Qwen3-TTS GUI - Tkinter 图形界面
调用 api-qwen.py 的 API 进行语音克隆合成
"""

import io
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import requests
import soundfile as sf
import sounddevice as sd


LANGUAGES = [
    ("Chinese", "中文"),
    ("English", "英文"),
    ("Japanese", "日文"),
    ("Korean", "韩文"),
    ("German", "德文"),
    ("French", "法文"),
    ("Russian", "俄文"),
    ("Portuguese", "葡萄牙语"),
    ("Spanish", "西班牙语"),
    ("Italian", "意大利语"),
    ("Auto", "自动检测"),
]

DEFAULT_API_URL = "http://127.0.0.1:9880"


class QwenTTSGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Qwen3-TTS 语音克隆")
        self.root.geometry("720x600")
        self.root.minsize(600, 500)
        self.root.resizable(True, True)

        self.ref_audio_path = tk.StringVar()
        self.api_url = tk.StringVar(value=DEFAULT_API_URL)

        self.audio_data = None
        self.audio_sr = None
        self.is_playing = False
        self.play_thread = None
        self._cancel_stream = False

        # True streaming state
        self._audio_queue = None           # queue.Queue[np.ndarray|None]
        self._gen_active = False           # generation thread running?
        self._gen_stop = threading.Event()
        self._output_stream = None         # sd.OutputStream
        self._chunk_data = None            # current chunk in callback
        self._chunk_pos = 0                # position in current chunk
        self._stream_lock = threading.Lock() # 用于保护流资源的锁

        self.lora_info = None
        self.lora_var = tk.StringVar()
        self.speaker_var = tk.StringVar()

        self._build_ui()
        self._update_play_btn()

    # ---------- UI ----------

    def _build_ui(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)

        # Row 0: Server config
        server_frame = ttk.LabelFrame(main, text="服务器设置", padding="5")
        server_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(server_frame, text="API 地址:").pack(side=tk.LEFT, padx=(0, 4))
        self.url_entry = ttk.Entry(server_frame, textvariable=self.api_url, width=30)
        self.url_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        self._make_tooltip(self.url_entry, "api-qwen.py 服务的地址，默认 http://127.0.0.1:9880")

        self.test_btn = ttk.Button(server_frame, text="测试连接", command=self._test_connection, width=10)
        self.test_btn.pack(side=tk.RIGHT)

        # Row 1: Reference audio
        ref_frame = ttk.LabelFrame(main, text="参考音频（语音克隆源）", padding="5")
        ref_frame.pack(fill=tk.X, pady=(0, 8))

        row1 = ttk.Frame(ref_frame)
        row1.pack(fill=tk.X)
        ttk.Label(row1, text="音频文件:").pack(side=tk.LEFT, padx=(0, 4))
        self.ref_entry = ttk.Entry(row1, textvariable=self.ref_audio_path)
        self.ref_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(row1, text="浏览...", command=self._browse_ref_audio, width=8).pack(side=tk.RIGHT)

        ttk.Label(ref_frame, text="参考文本 (可选, 不填则仅克隆音色):").pack(anchor=tk.W, pady=(6, 0))
        self.ref_text = tk.Text(ref_frame, height=2, wrap=tk.WORD)
        self.ref_text.pack(fill=tk.X, pady=(2, 0))
        self._make_tooltip(self.ref_text, "参考音频对应的文字内容。填写后采用 ICL 模式（效果更好），不填则采用 x_vector_only 模式（仅克隆音色）")

        # Row 2: Synthesis
        syn_frame = ttk.LabelFrame(main, text="合成设置", padding="5")
        syn_frame.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(syn_frame, text="合成文本:").pack(anchor=tk.W)
        self.syn_text = tk.Text(syn_frame, height=3, wrap=tk.WORD)
        self.syn_text.pack(fill=tk.X, pady=(2, 4))
        self._make_tooltip(self.syn_text, "要合成语音的目标文本")

        ctrl_row = ttk.Frame(syn_frame)
        ctrl_row.pack(fill=tk.X)

        ttk.Label(ctrl_row, text="语言:").pack(side=tk.LEFT, padx=(0, 4))
        self.lang_var = tk.StringVar(value="Chinese")
        self.lang_cb = ttk.Combobox(ctrl_row, textvariable=self.lang_var, values=[v[0] for v in LANGUAGES], state="readonly", width=14)
        self.lang_cb.pack(side=tk.LEFT, padx=(0, 16))
        self._make_tooltip(self.lang_cb, "合成文本的语言")

        ttk.Label(ctrl_row, text="切分符号:").pack(side=tk.LEFT, padx=(0, 4))
        self.cut_punc_var = tk.StringVar(value="")
        cut_cb = ttk.Combobox(ctrl_row, textvariable=self.cut_punc_var, values=["", "，。！？", ",.!?", "，。！？,.!?"], width=12)
        cut_cb.pack(side=tk.LEFT, padx=(0, 8))
        self._make_tooltip(cut_cb, "非流式模式下按符号拆分长文本分段合成，留空则不拆分")

        self.stream_var = tk.BooleanVar(value=True)
        self.stream_cb = ttk.Checkbutton(ctrl_row, text="流式播放", variable=self.stream_var)
        self.stream_cb.pack(side=tk.LEFT, padx=(0, 8))
        self._make_tooltip(self.stream_cb, "启用真流式：生成过程中即可实时播放，边生成边听")

        ttk.Label(ctrl_row, text="块:").pack(side=tk.LEFT, padx=(0, 2))
        self.chunk_var = tk.IntVar(value=8)
        chunk_spin = ttk.Spinbox(ctrl_row, from_=2, to=48, textvariable=self.chunk_var, width=4)
        chunk_spin.pack(side=tk.LEFT, padx=(0, 16))
        self._make_tooltip(chunk_spin, "流式块大小 (codec步数, 2~48, 越小延迟越低)")

        ttk.Label(ctrl_row, text="温度:").pack(side=tk.LEFT, padx=(0, 2))
        self.temp_var = tk.DoubleVar(value=0.9)
        temp_scale = ttk.Scale(ctrl_row, from_=0.1, to=2.0, variable=self.temp_var, length=80)
        temp_scale.pack(side=tk.LEFT, padx=(0, 2))
        self.temp_label = ttk.Label(ctrl_row, text="0.9", width=4)
        self.temp_label.pack(side=tk.LEFT, padx=(0, 16))
        temp_scale.configure(command=self._on_temp_change)
        self._make_tooltip(temp_scale, "采样温度，越高越随机 (0.1-2.0)")

        ttk.Label(ctrl_row, text="Top-K:").pack(side=tk.LEFT, padx=(0, 2))
        self.topk_var = tk.IntVar(value=50)
        topk_spin = ttk.Spinbox(ctrl_row, from_=1, to=200, textvariable=self.topk_var, width=5)
        topk_spin.pack(side=tk.LEFT, padx=(0, 16))

        ttk.Label(ctrl_row, text="Top-P:").pack(side=tk.LEFT, padx=(0, 2))
        self.topp_var = tk.DoubleVar(value=1.0)
        topp_spin = ttk.Spinbox(ctrl_row, from_=0.1, to=1.0, increment=0.05, textvariable=self.topp_var, width=5)
        topp_spin.pack(side=tk.LEFT)

        # Row 3: LoRA settings
        lora_frame = ttk.LabelFrame(main, text="LoRA 设置", padding="5")
        lora_frame.pack(fill=tk.X, pady=(0, 8))

        lora_row1 = ttk.Frame(lora_frame)
        lora_row1.pack(fill=tk.X)

        ttk.Label(lora_row1, text="选择 LoRA:").pack(side=tk.LEFT, padx=(0, 4))
        self.lora_cb = ttk.Combobox(lora_row1, textvariable=self.lora_var, state="readonly", width=22)
        self.lora_cb.pack(side=tk.LEFT, padx=(0, 8))
        self.lora_cb['values'] = ["不使用 LoRA"]
        self.lora_cb.bind("<<ComboboxSelected>>", self._on_lora_selected)

        ttk.Label(lora_row1, text="说话人:").pack(side=tk.LEFT, padx=(0, 4))
        self.speaker_cb = ttk.Combobox(lora_row1, textvariable=self.speaker_var, state="disabled", width=16)
        self.speaker_cb.pack(side=tk.LEFT, padx=(0, 8))
        self.speaker_cb['values'] = ["语音克隆模式"]

        self.refresh_lora_btn = ttk.Button(lora_row1, text="刷新列表", command=self._refresh_lora_list, width=10)
        self.refresh_lora_btn.pack(side=tk.RIGHT)

        ttk.Label(lora_frame, text="提示: 需服务端启用 --lora-all; 不选则使用参考音色克隆").pack(anchor=tk.W, pady=(2, 0))

        # Row 4: Buttons
        btn_frame = ttk.Frame(main)
        btn_frame.pack(fill=tk.X, pady=(0, 8))

        self.gen_btn = ttk.Button(btn_frame, text="生成语音", command=self._generate, width=14)
        self.gen_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.play_btn = ttk.Button(btn_frame, text="▶ 播放", command=self._play_audio, width=10)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.stop_btn = ttk.Button(btn_frame, text="■ 停止", command=self._stop_audio, width=8, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.save_btn = ttk.Button(btn_frame, text="保存音频...", command=self._save_audio, width=12)
        self.save_btn.pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate", length=120)
        self.progress.pack(side=tk.RIGHT, padx=(8, 0))

        # Row 4: Status + log
        status_frame = ttk.LabelFrame(main, text="状态", padding="5")
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(status_frame, height=6, wrap=tk.WORD, state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=scrollbar.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_temp_change(self, val):
        self.temp_label.configure(text=f"{float(val):.1f}")

    def _make_tooltip(self, widget, text):
        tip = None

        def show(event):
            nonlocal tip
            if tip:
                return
            x = event.x_root + 15
            y = event.y_root + 10
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{x}+{y}")
            lbl = ttk.Label(tip, text=text, background="#ffffc8", relief=tk.SOLID, borderwidth=1, padding=3)
            lbl.pack()

        def hide(event):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")

    # ---------- Actions ----------

    def _log(self, msg, level="info"):
        self.status_text.configure(state=tk.NORMAL)
        self.status_text.insert(tk.END, f"[{level.upper()}] {msg}\n")
        self.status_text.see(tk.END)
        self.status_text.configure(state=tk.DISABLED)

    def _test_connection(self):
        self._log("测试 API 连接...")
        self.test_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._do_test_connection, daemon=True).start()

    def _do_test_connection(self):
        try:
            r = requests.get(f"{self.api_url.get().rstrip('/')}/control?command=", timeout=5)
            if r.status_code == 200:
                self.root.after(0, lambda: self._log("API 连接成功", "ok"))
            else:
                self.root.after(0, lambda: self._log(f"API 返回状态码: {r.status_code}", "warn"))
        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: self._log("无法连接到 API 服务，请确认 api-qwen.py 已启动", "error"))
        except Exception as e:
            self.root.after(0, lambda: self._log(f"连接失败: {e}", "error"))
        finally:
            self.root.after(0, lambda: self.test_btn.configure(state=tk.NORMAL))

    # ---------- LoRA ----------

    def _on_lora_selected(self, event=None):
        lora_id = self.lora_var.get()
        if not lora_id or lora_id == "不使用 LoRA":
            self.speaker_cb['values'] = ["语音克隆模式"]
            self.speaker_cb.set("语音克隆模式")
            self.speaker_cb.configure(state=tk.DISABLED)
            return

        if self.lora_info and lora_id in self.lora_info.get("loras", {}):
            speakers = self.lora_info["loras"][lora_id].get("speakers", [])
            self.speaker_cb['values'] = ["语音克隆模式"] + speakers
            self.speaker_cb.set("语音克隆模式")
            self.speaker_cb.configure(state=tk.NORMAL)

    def _refresh_lora_list(self):
        self._log("正在刷新 LoRA 列表...")
        self.refresh_lora_btn.configure(state=tk.DISABLED)
        threading.Thread(target=self._do_refresh_lora_list, daemon=True).start()

    def _do_refresh_lora_list(self):
        try:
            r = requests.get(f"{self.api_url.get().rstrip('/')}/lora/list", timeout=5)
            if r.status_code == 200:
                data = r.json()
                self.lora_info = data
                self.root.after(0, self._update_lora_ui)
            else:
                self.root.after(0, lambda: self._log("多 LoRA 功能未启用 (服务端需 --lora-all)", "warn"))
        except Exception as e:
            self.root.after(0, lambda: self._log(f"刷新 LoRA 列表失败: {e}", "error"))
        finally:
            self.root.after(0, lambda: self.refresh_lora_btn.configure(state=tk.NORMAL))

    def _update_lora_ui(self):
        if not self.lora_info:
            return
        loras = list(self.lora_info.get("loras", {}).keys())
        self.lora_cb['values'] = ["不使用 LoRA"] + loras

        active = self.lora_info.get("active_lora_id")
        active_speaker = self.lora_info.get("active_speaker")

        if active and active in self.lora_info["loras"]:
            self.lora_var.set(active)
            speakers = self.lora_info["loras"][active].get("speakers", [])
            self.speaker_cb['values'] = ["语音克隆模式"] + speakers
            if active_speaker and active_speaker in speakers:
                self.speaker_var.set(active_speaker)
            else:
                self.speaker_var.set("语音克隆模式")
            self.speaker_cb.configure(state=tk.NORMAL)
            self._log(f"当前活跃 LoRA: {active}" + (f" ({active_speaker})" if active_speaker else ""), "info")
        else:
            self.lora_var.set("不使用 LoRA")
            self.speaker_cb['values'] = ["语音克隆模式"]
            self.speaker_cb.set("语音克隆模式")
            self.speaker_cb.configure(state=tk.DISABLED)
            self._log("当前无活跃 LoRA (voice_clone 模式)", "info")

    def _browse_ref_audio(self):
        path = filedialog.askopenfilename(
            title="选择参考音频",
            filetypes=[("音频文件", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("所有文件", "*.*")],
        )
        if path:
            self.ref_audio_path.set(path)

    def _generate(self):
        ref_path = self.ref_audio_path.get().strip()
        ref_text = self.ref_text.get("1.0", tk.END).strip()
        syn_text = self.syn_text.get("1.0", tk.END).strip()

        lora_id = self.lora_var.get().strip() if self.lora_info else ""
        has_speaker = (lora_id and lora_id != "不使用 LoRA"
                       and self.speaker_var.get().strip()
                       and self.speaker_var.get().strip() != "语音克隆模式")

        if not syn_text:
            messagebox.showwarning("缺少参数", "请输入合成文本")
            return
        if not ref_path and not has_speaker:
            messagebox.showwarning("缺少参数", "请选择参考音频文件")
            return

        # Reset state
        self._cancel_stream = False
        self._gen_stop.clear()
        self._gen_active = True
        self.audio_data = None
        self.audio_sr = None
        self._audio_queue = queue.Queue(maxsize=30)

        self.gen_btn.configure(state=tk.DISABLED)
        self.play_btn.configure(state=tk.DISABLED)
        self.save_btn.configure(state=tk.DISABLED)
        self.progress.start(10)
        self._log("正在生成语音...")

        use_stream = self.stream_var.get()
        lora_id = self.lora_var.get().strip() if self.lora_info else ""
        speaker = self.speaker_var.get().strip() if self.lora_info else ""

        payload = {
            "refer_wav_path": ref_path,
            "text": syn_text,
            "text_language": self.lang_var.get(),
            "top_k": self.topk_var.get(),
            "top_p": self.topp_var.get(),
            "temperature": self.temp_var.get(),
        }
        if ref_text:
            payload["prompt_text"] = ref_text

        if lora_id and lora_id != "不使用 LoRA":
            payload["lora_id"] = lora_id
            if speaker and speaker != "语音克隆模式":
                payload["speaker"] = speaker
        else:
            payload["lora_id"] = ""

        if use_stream:
            payload["stream_mode"] = "normal"
            payload["media_type"] = "ogg"
            payload["chunk_size"] = self.chunk_var.get()
        else:
            # Non-streaming: still use cut_punc text splitting
            payload["cut_punc"] = self.cut_punc_var.get()

        threading.Thread(target=self._do_generate, args=(payload, use_stream), daemon=True).start()

    def _do_generate(self, payload, use_stream):
        try:
            url = self.api_url.get().rstrip("/") + "/"

            if use_stream:
                self._do_generate_stream(url, payload)
            else:
                self._do_generate_normal(url, payload)
        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: self._log("无法连接到 API 服务", "error"))
        except Exception as e:
            self.root.after(0, lambda: self._log(f"生成异常: {e}", "error"))
        finally:
            self._gen_active = False
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.gen_btn.configure(state=tk.NORMAL))
            self.root.after(0, lambda: self.save_btn.configure(state=tk.NORMAL))
            self.root.after(0, self._update_play_btn)

    def _do_generate_normal(self, url, payload):
        r = requests.post(url, json=payload, timeout=300)
        if r.status_code == 200 and len(r.content) > 0:
            data, sr = sf.read(io.BytesIO(r.content))
            self.audio_data = data
            self.audio_sr = sr
            duration = len(data) / sr if sr else 0
            self.root.after(0, lambda: self._log(f"生成成功！时长 {duration:.1f}s, 采样率 {sr}Hz", "ok"))
        else:
            self._handle_error(r)

    def _do_generate_stream(self, url, payload):
        """Producer: fetch streaming audio, decode OGG chunks, feed queue."""
        r = requests.post(url, json=payload, stream=True, timeout=300)
        if r.status_code != 200:
            self._handle_error(r)
            self._audio_queue.put(None)
            return

        chunks = []
        segments = 0
        try:
            for raw in r.iter_content(chunk_size=None):
                if self._gen_stop.is_set():
                    r.close()
                    self.root.after(0, lambda: self._log("流式生成已取消", "warn"))
                    self._audio_queue.put(None)
                    return
                if not raw:
                    continue
                try:
                    data, sr = sf.read(io.BytesIO(raw))
                    chunks.append(data)
                    segments += 1
                    if self.audio_sr is None:
                        self.audio_sr = sr

                    # Feed queue for real-time playback
                    try:
                        self._audio_queue.put(data, timeout=2)
                    except queue.Full:
                        pass

                    # Accumulate final audio (lock-free single-writer OK)
                    combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                    self.audio_data = combined
                    self.root.after(0, lambda s=segments: self._log(f"片段 {s} (队列:{self._audio_queue.qsize()})", "ok"))
                    self.root.after(0, self._update_play_btn)
                except Exception as e:
                    self.root.after(0, lambda: self._log(f"解析音频片段失败: {e}", "warn"))
        finally:
            self._audio_queue.put(None)  # sentinel: generation done

        if chunks:
            duration = len(self.audio_data) / self.audio_sr if self.audio_sr else 0
            self.root.after(0, lambda: self._log(f"流式生成完成！{segments} 片段, 总时长 {duration:.1f}s", "ok"))

    def _handle_error(self, r):
        try:
            err = r.json()
            msg = err.get("message", r.text)
        except Exception:
            msg = r.text or f"HTTP {r.status_code}"
        self.root.after(0, lambda: self._log(f"生成失败: {msg}", "error"))

    def _audio_callback(self, outdata, frames, time_info, status):
        """Called by sounddevice when more audio data is needed."""
        filled = 0
        while filled < frames:
            if self._chunk_data is None or self._chunk_pos >= len(self._chunk_data):
                try:
                    # 使用 get_nowait 防止回调阻塞
                    new_chunk = self._audio_queue.get_nowait() if self._audio_queue else None
                except queue.Empty:
                    outdata[filled:] = 0
                    return

                if new_chunk is None:  # sentinel
                    outdata[filled:] = 0
                    raise sd.CallbackStop()
                self._chunk_data = new_chunk
                self._chunk_pos = 0

            avail = len(self._chunk_data) - self._chunk_pos
            need = frames - filled
            take = min(avail, need)

            outdata[filled:filled+take, 0] = self._chunk_data[self._chunk_pos:self._chunk_pos+take]
            self._chunk_pos += take
            filled += take

    def _play_audio(self):
        if self.is_playing:
            self._pause_playback()
            return

        if self._gen_active and self._audio_queue is not None:
            self._start_streaming_playback()
        elif self.audio_data is not None:
            self._start_normal_playback()

    def _pause_playback(self):
        """Pause playback safely using locks."""
        self._log("正在暂停...", "info")
        with self._stream_lock:
            try:
                if self._output_stream is not None:
                    self._output_stream.stop()
                    self._output_stream.close()
                    self._output_stream = None
            except Exception:
                pass
        self.is_playing = False
        self._update_play_btn()
        self.stop_btn.configure(state=tk.DISABLED)
        self._log("已暂停", "info")

    def _start_streaming_playback(self):
        """Consumer: play audio chunks from queue via OutputStream callback."""
        if self.audio_sr is None:
            self._log("采样率未就绪，请稍后重试", "warn")
            return

        self.is_playing = True
        self._update_play_btn()
        self.stop_btn.configure(state=tk.NORMAL)
        self._log("流式播放中...", "ok")

        def _stream_player():
            with self._stream_lock:
                try:
                    # 移除 finished_callback，改用线程轮询等待，避免回调触发死锁
                    self._output_stream = sd.OutputStream(
                        samplerate=self.audio_sr,
                        channels=1,
                        callback=self._audio_callback,
                    )
                    self._output_stream.start()
                except Exception as e:
                    self.root.after(0, lambda: self._log(f"播放启动失败: {e}", "error"))
                    self.root.after(0, self._cleanup_stream)
                    return

            # 在锁外面等待流播放完毕，不阻塞 stop 按钮的响应
            try:
                while True:
                    with self._stream_lock:
                        if self._output_stream is None or not self._output_stream.active:
                            break
                    sd.sleep(100)
            except Exception:
                pass
            finally:
                # 正常播放完毕后通过主线程队列安全关闭
                self.root.after(0, self._cleanup_stream)

        self.play_thread = threading.Thread(target=_stream_player, daemon=True)
        self.play_thread.start()

    def _start_normal_playback(self):
        """Play pre-generated audio via sd.play (blocking)."""
        self.is_playing = True
        self._update_play_btn()
        self.stop_btn.configure(state=tk.NORMAL)

        data = self.audio_data.copy()
        sr = self.audio_sr

        def _play():
            try:
                sd.play(data, sr)
                sd.wait()
            except Exception:
                pass
            finally:
                self.is_playing = False
                self.root.after(0, self._update_play_btn)
                self.root.after(0, lambda: self.stop_btn.configure(state=tk.DISABLED))

        self.play_thread = threading.Thread(target=_play, daemon=True)
        self.play_thread.start()

    def _cleanup_stream(self):
        """主线程调用的安全清理流函数"""
        with self._stream_lock:
            try:
                if self._output_stream is not None:
                    self._output_stream.stop()
                    self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None
        self.is_playing = False
        self._update_play_btn()
        self.stop_btn.configure(state=tk.DISABLED)

    def _stop_audio(self):
        self._gen_stop.set()
        self._cancel_stream = True

        # 加锁安全释放 sounddevice 资源
        with self._stream_lock:
            try:
                if self._output_stream is not None:
                    self._output_stream.stop()
                    self._output_stream.close()
                    self._output_stream = None
            except Exception:
                pass
        try:
            sd.stop()
        except Exception:
            pass

        # 清空队列
        if self._audio_queue is not None:
            try:
                while True:
                    self._audio_queue.get_nowait()
            except queue.Empty:
                pass

        self._chunk_data = None
        self._chunk_pos = 0
        self.is_playing = False
        self._update_play_btn()
        self.stop_btn.configure(state=tk.DISABLED)
        self._log("已完全停止", "info")

    def _save_audio(self):
        if self.audio_data is None:
            messagebox.showwarning("无音频", "请先生成语音")
            return
        path = filedialog.asksaveasfilename(
            title="保存音频",
            defaultextension=".wav",
            filetypes=[("WAV 文件", "*.wav"), ("所有文件", "*.*")],
        )
        if path:
            sf.write(path, self.audio_data, self.audio_sr)
            self._log(f"已保存: {os.path.basename(path)}", "ok")

    def _update_play_btn(self):
        can_play = (
            self.audio_data is not None
            or (self._gen_active and self._audio_queue is not None and not self._audio_queue.empty())
        )
        if can_play:
            self.play_btn.configure(state=tk.NORMAL)
            self.play_btn.configure(text="⏸ 暂停" if self.is_playing else "▶ 播放")
        else:
            self.play_btn.configure(state=tk.DISABLED)
            self.play_btn.configure(text="▶ 播放")


def main():
    root = tk.Tk()
    app = QwenTTSGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()