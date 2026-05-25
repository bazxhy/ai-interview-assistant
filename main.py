#!/usr/bin/env python3
"""
AI 面试助手 - 实时监听面试音频，AI 智能解答
============================================

功能:
  - 实时麦克风监听，自动检测语音段落
  - 语音转文字 (OpenAI Whisper API / 本地 Whisper)
  - 多轮对话理解 + AI 智能回答生成
  - 支持 OpenAI / DeepSeek / 任何兼容 OpenAI 接口的 LLM

用法:
  1. cp .env.example .env        # 配置 API Key
  2. pip install -r requirements.txt
  3. python main.py

快捷键:
  Space    手动触发录音 (处理最近 5 秒音频)
  C        清除对话历史
  Q        退出程序
  Ctrl+C   直接退出
"""

import os
import sys
import time
import wave
import json
import ssl
import hmac
import hashlib
import base64
import shutil
import signal
import argparse
import tempfile
import threading
import traceback
from urllib.parse import urlencode
from pathlib import Path
from queue import Queue, Empty
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict

import numpy as np

# ============================================================
# 依赖检查 & 动态加载
# ============================================================

try:
    import sounddevice as sd
except ImportError:
    print("[错误] 缺少 sounddevice 库, 请运行: pip install sounddevice")
    sys.exit(1)

# VAD 使用基于能量的自实现方案，无需额外依赖

try:
    from openai import OpenAI
except ImportError:
    print("[错误] 缺少 openai 库, 请运行: pip install openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("[警告] 缺少 python-dotenv, 将使用环境变量或默认配置")
    def load_dotenv(*args, **kwargs):
        pass

_HAS_KEYBOARD = False
try:
    import keyboard
    _HAS_KEYBOARD = True
except ImportError:
    _HAS_KEYBOARD = False

_HAS_TTS = False
try:
    import pyttsx3
    _HAS_TTS = True
except ImportError:
    pass

_HAS_FASTER_WHISPER = False
try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    _HAS_FASTER_WHISPER = True
except ImportError:
    pass

# ============================================================
# 配置
# ============================================================

load_dotenv(override=True)

# --- 音频 ---
SAMPLE_RATE       = 16000
FRAME_DURATION_MS = 30                     # VAD 帧长度 (10/20/30 ms)
CHANNELS          = 1
BLOCK_SIZE        = 1024                   # 每次回调采样数
_DEV = int(os.getenv("AUDIO_DEVICE", "-1"))
DEVICE_INDEX      = None if _DEV < 0 else _DEV  # None=默认设备
AUDIO_MODE        = os.getenv("AUDIO_MODE", "mic")  # mic=麦克风, loopback=系统声音(面试官)

# --- VAD ---
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0"))   # 0=自动校准
VAD_ADAPTIVE         = os.getenv("VAD_ADAPTIVE", "true").lower() != "false"
SILENCE_DURATION     = float(os.getenv("SILENCE_DURATION", "1.5"))
MIN_SPEECH_DURATION  = float(os.getenv("MIN_SPEECH_DURATION", "0.5"))

# --- STT ---
STT_PROVIDER = os.getenv("STT_PROVIDER", "openai")
STT_MODEL    = os.getenv("STT_MODEL", "whisper-1")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "zh")

# --- 讯飞 ---
IFLYTEK_APP_ID     = os.getenv("IFLYTEK_APP_ID", "")
IFLYTEK_API_KEY    = os.getenv("IFLYTEK_API_KEY", "")
IFLYTEK_API_SECRET = os.getenv("IFLYTEK_API_SECRET", "")

# --- LLM ---
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL       = os.getenv("LLM_MODEL", "gpt-4o")
LLM_API_KEY     = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
LLM_API_BASE    = os.getenv("OPENAI_API_BASE", "") or ""
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# 自动修正 DeepSeek 的 API 地址
if LLM_PROVIDER == "deepseek" and not LLM_API_BASE:
    LLM_API_BASE = "https://api.deepseek.com"
if LLM_PROVIDER == "deepseek" and not LLM_MODEL:
    LLM_MODEL = "deepseek-chat"

# --- 功能开关 ---
ENABLE_TTS  = os.getenv("ENABLE_TTS", "false").lower() == "true"
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "5"))

# --- 简历自动检测与读取 ---
_RESUME_TEXT: str = ""

def _find_and_read_resume() -> str:
    """扫描当前目录, 找到简历文件并提取文本"""
    import glob as _glob
    patterns = ["*简历*", "*resume*", "*cv*", "*.pdf", "*.txt"]
    seen = set()
    for pat in patterns:
        for f in _glob.glob(pat):
            f_lower = f.lower()
            if f_lower in seen:
                continue
            seen.add(f_lower)
            try:
                if f.endswith(".pdf"):
                    try:
                        import pdfplumber
                        with pdfplumber.open(f) as pdf:
                            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    except ImportError:
                        continue
                elif f.endswith(".txt"):
                    with open(f, "r", encoding="utf-8") as fh:
                        text = fh.read()
                else:
                    continue
                if text and len(text.strip()) > 50:
                    print_ok(f"读取简历: {f} ({len(text)} 字)")
                    return text.strip()
            except Exception:
                continue
    return ""

# --- 系统提示词 ---
_SYSTEM_PROMPT_BASE = os.getenv("SYSTEM_PROMPT", "").strip()
if not _SYSTEM_PROMPT_BASE:
    _SYSTEM_PROMPT_BASE = """你是一个专业的面试助手，正在帮助面试者实时回答面试官提出的技术问题。

请严格遵循以下规则:
1. 回答简洁有力，控制在 200-400 字以内，方便面试者记忆和复述
2. 技术问题务必准确，代码示例尽量简短
3. 行为类问题用 STAR 法则: 情境-任务-行动-结果
4. 语言自然流畅，像真人说话一样
5. 如果问题不完整或模糊，请指出并请求澄清
6. 如果上一轮对话中已经回答过相关内容，可以适当引用"""

SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE  # 默认值, main() 中加载简历后更新

# --- 本地 Whisper 模型 ---
LOCAL_WHISPER_MODEL = os.getenv("LOCAL_WHISPER_MODEL", "base")
LOCAL_WHISPER_DEVICE = os.getenv("LOCAL_WHISPER_DEVICE", "cpu")

# --- 关键词纠错 ---
_KEYWORD_MAP: Dict[str, str] = {}
def _load_keywords(filepath: str = "keywords.txt") -> Dict[str, str]:
    """加载关键词纠错表。格式: 正确词,误识别1,误识别2,..."""
    mapping: Dict[str, str] = {}
    if not os.path.exists(filepath):
        return mapping
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) >= 2:
                correct = parts[0]
                for alt in parts[1:]:
                    mapping[alt.lower()] = correct
            elif len(parts) == 1:
                mapping[parts[0].lower()] = parts[0]
    return mapping

def correct_keywords(text: str) -> str:
    """用关键词表修正语音识别结果中的专业术语"""
    if not _KEYWORD_MAP:
        return text
    result = text
    # 按误识别字符串长度降序排列，优先匹配长的
    for wrong in sorted(_KEYWORD_MAP.keys(), key=len, reverse=True):
        if wrong.lower() in result.lower():
            # 全词匹配替换
            result = result.replace(wrong, _KEYWORD_MAP[wrong])
    return result

# 加载关键词表
_KEYWORD_MAP = _load_keywords("keywords.txt")

# --- 快速回答模板 ---
_SELF_INTRO = os.getenv("SELF_INTRO", "")

def _is_continuation(text: str) -> bool:
    """判断识别文本是否是前一句的补充 (而非新问题)"""
    t = text.strip()
    if len(t) <= 15:
        return True  # 短句大概率是补充 (如"具体都用了哪些机制")
    # 以连接性标点、连词开头 → 补充句
    cont_starts = ["？", "?", "、", "还有", "另外", "具体", "比如", "例如",
                   "特别是", "尤其是", "那", "那么", "就是", "也就是", "而且",
                   "然后", "接着", "并且", "以及", "或者说", "还是说"]
    for s in cont_starts:
        if t.startswith(s):
            return True
    # 以全新问句开头 → 新问题
    new_starts = ["你好", "您好", "先", "首先", "接下", "下一个", "另外",
                  "换一", "然后我们", "我们聊", "再说", "谈谈", "讲讲",
                  "你自我", "介绍", "你的期望", "薪资", "工资", "待遇"]
    for s in new_starts:
        if t.startswith(s):
            return False
    # 默认 → 合并 (宁可多拼也不错丢)
    return True

def _match_quick_response(question: str) -> Optional[str]:
    """匹配预设回答模板, 命中则直接返回免 AI 调用"""
    q = question.lower().replace(" ", "")
    # 自我介绍
    if any(kw in q for kw in ["自我介绍", "介绍自己", "介绍下自己", "说说你自己",
                                "介绍下你自己", "做个自我介绍", "简单介绍", "先自我介绍一下"]):
        if _SELF_INTRO:
            return _SELF_INTRO
    return None


# --- Q&A 知识库 ---
_QA_BANK: List[dict] = []
_QA_BANK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "qa_bank.json")

def _load_qa_bank() -> List[dict]:
    try:
        with open(_QA_BANK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

_QA_BANK = _load_qa_bank()

def _match_qa_bank(question: str, min_kw: int = 2, min_ratio: float = 0.25) -> Optional[str]:
    """在 Q&A 知识库中匹配面试官问题, 返回最佳预设回答或 None"""
    if not _QA_BANK or not question:
        return None
    q_lower = question.lower()

    best_score = 0.0
    best_answer: Optional[str] = None

    for entry in _QA_BANK:
        keywords = entry.get("keywords", [])
        if not keywords:
            continue
        hits = sum(1 for kw in keywords if kw.lower() in q_lower)
        ratio = hits / len(keywords)

        # 综合评分: 命中关键词数 * 0.4 + 命中比例 * 0.6
        score = hits * 0.4 + ratio * 0.6

        if hits >= min_kw and ratio >= min_ratio and score > best_score:
            best_score = score
            best_answer = entry.get("answer", "")

    return best_answer

# ============================================================
# 终端输出工具
# ============================================================

class Colors:
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE    = '\033[94m'
    BOLD    = '\033[1m'
    RESET   = '\033[0m'

def supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    if supports_color():
        return f"{code}{text}{Colors.RESET}"
    return text

def print_sep(char: str = "─", width: int = 56):
    print(char * width)

def print_info(msg: str):
    print(_c(Colors.CYAN, f"[*] {msg}"))

def print_ok(msg: str):
    print(_c(Colors.GREEN, f"[+] {msg}"))

def print_warn(msg: str):
    print(_c(Colors.YELLOW, f"[!] {msg}"))

def print_err(msg: str):
    print(_c(Colors.RED, f"[x] {msg}"))

def print_q(interviewer_question: str, ai_answer: str = "", stream_mode: bool = False):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(_c(Colors.BOLD, f"  [{timestamp}]") + _c(Colors.YELLOW, f" 面试官: {interviewer_question}"))
    print_sep("-")
    if stream_mode:
        # 流式模式下 AI 回答已逐字打印, 只显示分隔
        print(_c(Colors.GREEN, "  AI 建议 (流式): 见上方 ↑"))
    else:
        print(_c(Colors.GREEN, f"  AI 建议: {ai_answer}"))
    print(f"{'='*60}")

# ============================================================
# 音频处理器
# ============================================================

class AudioStream:
    """音频采集 + 能量型语音活动检测 (Energy-based VAD)"""

    # 流式队列的哨兵值
    STREAM_START = object()
    STREAM_END   = object()

    def __init__(self):
        self.frame_samples     = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
        self.frame_bytes       = self.frame_samples * 2
        self.energy_threshold  = VAD_ENERGY_THRESHOLD             # 0 = 自动校准
        self.noise_floor       = 3.0                              # 初始噪声估计 (RMS)

        self.silence_frames = max(1, int(SILENCE_DURATION * 1000 / FRAME_DURATION_MS))
        self.min_frames     = max(1, int(MIN_SPEECH_DURATION * 1000 / FRAME_DURATION_MS))

        # 环形缓冲
        ring_secs = max(2.0, MIN_SPEECH_DURATION + 1.0)
        max_blocks = max(1, int(ring_secs * SAMPLE_RATE / BLOCK_SIZE))
        self.ring_buffer = deque(maxlen=max_blocks)

        # 状态
        self.is_speaking    = False
        self.speech_buffer: List[bytes] = []
        self.silence_count  = 0
        self.speech_count   = 0

        self._noise_buffer: List[float] = []   # 用于校准噪声水平
        self._calibrated     = VAD_ENERGY_THRESHOLD > 0

        self._leftover = b""
        self.segment_queue: Queue = Queue()   # 完整语音段 (兼容批量模式)
        self.stream_queue: Queue = Queue()    # 实时音频帧 (流式模式)
        self._stream: Optional[sd.InputStream] = None

    # ---------- 能量计算 ----------

    @staticmethod
    def _rms(frame_bytes: bytes) -> float:
        arr = np.frombuffer(frame_bytes, dtype=np.int16).astype(np.float64)
        return float(np.sqrt(np.mean(arr ** 2)))

    def _is_speech(self, frame_bytes: bytes) -> bool:
        rms = self._rms(frame_bytes)

        if not self._calibrated and self.noise_floor > 0 and len(self._noise_buffer) < 300:
            self._noise_buffer.append(rms)
            if len(self._noise_buffer) == 300:
                self.noise_floor = max(float(np.mean(self._noise_buffer)), 1.0)
                if VAD_ADAPTIVE:
                    self.energy_threshold = self.noise_floor * 2.0
                    print_info(f"\nVAD 自动校准完成: 噪声 ~{self.noise_floor:.1f}, 阈值 ~{self.energy_threshold:.1f}")
                else:
                    self.energy_threshold = self.noise_floor * 2.5
                    print_info(f"\nVAD 校准完成 (固定模式): 噪声 ~{self.noise_floor:.1f}, 阈值 ~{self.energy_threshold:.1f}")
                self._calibrated = True
            return False

        if VAD_ADAPTIVE and self._calibrated:
            self.noise_floor = self.noise_floor * 0.98 + rms * 0.02
            self.energy_threshold = self.noise_floor * 2.0

        return rms > self.energy_threshold

    # ---------- 核心回调 ----------

    def _callback(self, indata: np.ndarray, frames: int, time_info, status):
        if status and status.input_overflow:
            print_warn(f"音频溢出: {status}")
            return

        audio_int16 = (indata[:, 0] * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        self.ring_buffer.append(audio_int16)

        data = self._leftover + audio_int16
        offset = 0
        total = len(data)

        while offset + self.frame_bytes <= total:
            frame = data[offset:offset + self.frame_bytes]
            offset += self.frame_bytes

            if self._is_speech(frame):
                self.speech_count += 1
                if self.speech_count >= 3 and not self.is_speaking:
                    self.is_speaking = True
                    self.stream_queue.put(AudioStream.STREAM_START)
                    padding_blocks = 3
                    for blk in list(self.ring_buffer)[-padding_blocks:]:
                        self.speech_buffer.append(blk)
                        self.stream_queue.put(blk)
                if self.is_speaking:
                    self.speech_buffer.append(frame)
                    self.stream_queue.put(frame)
                self.silence_count = 0
            else:
                if self.is_speaking:
                    self.speech_buffer.append(frame)
                    self.stream_queue.put(frame)
                    self.silence_count += 1
                    if self.silence_count >= self.silence_frames:
                        self.stream_queue.put(AudioStream.STREAM_END)
                        self._finish_segment()
                else:
                    self.speech_count = 0

        self._leftover = data[offset:]

    def _finish_segment(self):
        if len(self.speech_buffer) < self.min_frames:
            self._reset_state()
            return

        full_audio = b"".join(self.speech_buffer)
        self.segment_queue.put(full_audio)
        self._reset_state()

    def _reset_state(self):
        self.speech_buffer.clear()
        self.is_speaking = False
        self.silence_count = 0
        self.speech_count = 0

    # ---------- 公开接口 ----------

    def start(self):
        if AUDIO_MODE == "loopback":
            self._start_loopback()
        else:
            self._start_mic()

    def _start_mic(self):
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                device=DEVICE_INDEX,
                blocksize=BLOCK_SIZE,
                callback=self._callback,
                dtype=np.float32,
            )
            self._stream.start()
        except sd.PortAudioError as e:
            print_err(f"无法打开音频设备: {e}")
            print_info("可用设备列表:")
            try:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    print(f"  [{i}] {d['name']} (in={d['max_input_channels']}, out={d['max_output_channels']})")
            except Exception:
                pass
            raise

    def _start_loopback(self):
        """通过 WASAPI loopback 录制系统音频 (面试官声音)"""
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            print_err("loopback 模式需要 pyaudiowpatch: pip install pyaudiowpatch")
            raise

        self._pa = pyaudio.PyAudio()
        try:
            loopback = self._pa.get_default_wasapi_loopback()
        except Exception as e:
            print_err(f"找不到 WASAPI loopback 设备: {e}")
            self._pa.terminate()
            raise

        self._loopback_rate = int(loopback["defaultSampleRate"])
        self._loopback_ch = loopback["maxInputChannels"]
        self._loopback_buf: List[float] = []  # 积累单声道采样用于重采样

        print_ok(f"Loopback: {loopback['name']} ({self._loopback_rate}Hz, {self._loopback_ch}ch)")

        # 每 40ms 读取一次
        self._loopback_frames = int(self._loopback_rate * 0.04)
        self._loopback_running = True

        # 在后台线程中持续读取
        def _reader():
            with self._pa.open(
                format=pyaudio.paInt16,
                channels=self._loopback_ch,
                rate=self._loopback_rate,
                frames_per_buffer=self._loopback_frames,
                input=True,
                input_device_index=loopback["index"],
            ) as stream:
                while self._loopback_running:
                    try:
                        data = stream.read(self._loopback_frames, exception_on_overflow=False)
                    except Exception:
                        continue
                    self._loopback_process(data)

        self._loopback_thread = threading.Thread(target=_reader, daemon=True)
        self._loopback_thread.start()

    _lb_rms_acc = 0.0
    _lb_rms_count = 0
    _lb_print_every = 8  # 约每 8*40ms=320ms 刷新
    _lb_pause = False    # AI 流式输出时暂停显示

    def _loopback_process(self, data: bytes):
        """将 loopback 音频 (立体声, 原生采样率) 转为 16kHz mono int16,
        并送入 VAD 流水线"""
        # stereo → mono
        arr = np.frombuffer(data, dtype=np.int16).astype(np.float64)
        arr = arr.reshape(-1, self._loopback_ch)
        mono = arr.mean(axis=1)

        # 实时系统声音电平
        self._lb_rms_acc += float(np.sqrt(np.mean(mono ** 2)))
        self._lb_rms_count += 1
        if self._lb_rms_count >= self._lb_print_every and not AudioStream._lb_pause:
            avg_rms = self._lb_rms_acc / self._lb_rms_count
            bar_len = min(20, max(1, int(avg_rms / 100)))
            bar = "#" * bar_len + " " * (20 - bar_len)
            print(f"\r  系统声音: {avg_rms:6.0f} [{bar}]  ", end="", flush=True)
            self._lb_rms_acc = 0.0
            self._lb_rms_count = 0

        # 重采样到 16kHz (线性插值)
        n_in = len(mono)
        duration = n_in / self._loopback_rate
        n_out = int(duration * SAMPLE_RATE)
        if n_out < 1:
            return
        x_in = np.linspace(0, duration, n_in, endpoint=False)
        x_out = np.linspace(0, duration, n_out, endpoint=False)
        resampled = np.interp(x_out, x_in, mono)

        # 接入现有 VAD 流水线 (模拟 sounddevice 回调)
        self._callback(
            resampled.reshape(-1, 1).astype(np.float32) / 32768.0,
            len(resampled), None, None,
        )

    def stop(self):
        if AUDIO_MODE == "loopback":
            self._loopback_running = False
            if hasattr(self, '_pa'):
                try:
                    self._pa.terminate()
                except Exception:
                    pass
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_recent(self, duration: float = 5.0) -> bytes:
        needed_blocks = int(duration * SAMPLE_RATE / BLOCK_SIZE)
        blocks = list(self.ring_buffer)[-needed_blocks:]
        return b"".join(blocks)


# ============================================================
# 语音转文字 (STT)
# ============================================================

class STTEngine:
    """语音转文字，支持 OpenAI Whisper API 和本地模型"""

    def __init__(self, openai_client: Optional[OpenAI] = None):
        self.client = openai_client
        self._local_model = None

    def transcribe(self, audio_bytes: bytes) -> str:
        if STT_PROVIDER == "local":
            return self._transcribe_local(audio_bytes)
        elif STT_PROVIDER == "xunfei":
            return self._transcribe_xunfei(audio_bytes)
        return self._transcribe_api(audio_bytes)

    def _transcribe_api(self, audio_bytes: bytes) -> str:
        if self.client is None:
            print_err("OpenAI 客户端未初始化，无法转写音频")
            return ""

        # 写临时 WAV
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)

            with open(wav_path, "rb") as f:
                kwargs = {"model": STT_MODEL, "file": f}
                if STT_LANGUAGE:
                    kwargs["language"] = STT_LANGUAGE
                result = self.client.audio.transcriptions.create(**kwargs)

            return result.text.strip()

        except Exception as e:
            print_err(f"STT 失败: {e}")
            return ""
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def _transcribe_local(self, audio_bytes: bytes) -> str:
        if not _HAS_FASTER_WHISPER:
            print_err("本地模式需要 faster-whisper: pip install faster-whisper")
            return ""
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            return ""

        if self._local_model is None:
            print_info(f"正在加载本地 Whisper 模型 ({LOCAL_WHISPER_MODEL})...")
            try:
                compute = "int8" if LOCAL_WHISPER_DEVICE == "cpu" else "float16"
                self._local_model = WhisperModel(
                    LOCAL_WHISPER_MODEL,
                    device=LOCAL_WHISPER_DEVICE,
                    compute_type=compute,
                )
                print_ok("本地模型加载完成")
            except Exception as e:
                print_err(f"加载本地模型失败: {e}")
                return ""

        # 写临时文件
        wav_path = None
        try:
            fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_bytes)

            segments, _ = self._local_model.transcribe(
                wav_path,
                language=STT_LANGUAGE if STT_LANGUAGE else None,
            )
            return " ".join(seg.text.strip() for seg in segments)

        except Exception as e:
            print_err(f"本地 STT 失败: {e}")
            return ""
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

    def _transcribe_xunfei(self, audio_bytes: bytes) -> str:
        """批量模式：一次性发送完整音频到讯飞"""
        q: Queue = Queue()
        q.put(AudioStream.STREAM_START)
        for offset in range(0, len(audio_bytes), 1280):
            q.put(audio_bytes[offset:offset + 1280])
        q.put(AudioStream.STREAM_END)
        return self.transcribe_streaming(q)

    def transcribe_streaming(self, audio_queue: Queue, started: bool = False) -> str:
        """流式模式：从队列读取音频帧，实时发送到讯飞，返回完整识别文本"""
        try:
            import websocket
        except ImportError:
            print_err("讯飞模式需要 websocket-client: pip install websocket-client")
            return ""

        url = _build_xunfei_url()
        ws: Optional[websocket.WebSocket] = None
        error_msg: List[str] = []

        DATA_FMT = "audio/L16;rate=16000"
        DATA_ENC = "raw"
        CHUNK = 1280

        if not started:
            while True:
                try:
                    if audio_queue.get(timeout=0.2) is AudioStream.STREAM_START:
                        break
                except Empty:
                    continue

        try:
            ws = websocket.create_connection(
                url, timeout=10,
                sslopt={"cert_reqs": ssl.CERT_NONE},
            )
        except Exception as e:
            print_err(f"讯飞连接失败: {e}")
            return ""

        acc = bytearray()
        first_sent = False
        ended = False
        all_fragments: List[str] = []
        last_text = ""

        while not ended:
            got_end = False
            while len(acc) < CHUNK:
                try:
                    item = audio_queue.get(timeout=0.03)
                except Empty:
                    break
                if item is AudioStream.STREAM_END:
                    got_end = True
                    break
                if item is not AudioStream.STREAM_START:
                    acc.extend(item)

            chunk = bytes(acc)

            if not first_sent and len(chunk) >= CHUNK:
                ws.send(json.dumps({
                    "common": {"app_id": IFLYTEK_APP_ID},
                    "business": {
                        "domain": "iat", "language": "zh_cn",
                        "accent": "mandarin", "ptt": 1,
                        "vad_eos": 2000, "dwa": "wpgs",
                    },
                    "data": {
                        "status": 0, "format": DATA_FMT,
                        "encoding": DATA_ENC,
                        "audio": base64.b64encode(chunk).decode(),
                    },
                }))
                acc.clear()
                first_sent = True
            elif first_sent and len(chunk) >= CHUNK:
                if got_end:
                    ws.send(json.dumps({
                        "data": {
                            "status": 2, "format": DATA_FMT,
                            "encoding": DATA_ENC,
                            "audio": base64.b64encode(chunk).decode(),
                        },
                    }))
                    ws.send(json.dumps({"data": {"status": 2}}))
                    ended = True
                else:
                    ws.send(json.dumps({
                        "data": {
                            "status": 1, "format": DATA_FMT,
                            "encoding": DATA_ENC,
                            "audio": base64.b64encode(chunk).decode(),
                        },
                    }))
                    acc.clear()
            elif got_end:
                if first_sent and len(chunk) > 0:
                    ws.send(json.dumps({
                        "data": {
                            "status": 2, "format": DATA_FMT,
                            "encoding": DATA_ENC,
                            "audio": base64.b64encode(chunk).decode(),
                        },
                    }))
                ws.send(json.dumps({"data": {"status": 2}}))
                ended = True

            if got_end:
                acc.clear()

            # 接收服务器消息 (非阻塞)
            old_timeout = ws.gettimeout()
            ws.settimeout(0.02)
            try:
                while True:
                    try:
                        msg = json.loads(ws.recv())
                    except websocket.WebSocketTimeoutException:
                        break
                    except Exception:
                        break
                    code = msg.get("code", 0)
                    if code != 0:
                        error_msg.append(f"code={code}: {msg.get('message', '')}")
                        ended = True
                        break
                    data = msg.get("data", {})
                    result = data.get("result", {})
                    text = "".join(cw.get("w", "") for ws_item in result.get("ws", [])
                                   for cw in ws_item.get("cw", []))
                    if text:
                        # 去重叠: 只在 last_text 后面追加新增部分
                        if text.startswith(last_text):
                            new_part = text[len(last_text):]
                        else:
                            new_part = text
                        if new_part:
                            all_fragments.append(new_part)
                            print(_c(Colors.MAGENTA, f"  -> {new_part}"))
                        last_text = text
            finally:
                try:
                    ws.settimeout(old_timeout)
                except Exception:
                    pass

        try:
            ws.close()
        except Exception:
            pass

        if error_msg:
            print_err(f"讯飞 STT: {'; '.join(error_msg)}")
        return "".join(all_fragments) if all_fragments else last_text


def _build_xunfei_url() -> str:
    """生成讯飞 IAT WebSocket 鉴权 URL"""
    from wsgiref.handlers import format_date_time

    host = "ws-api.xfyun.cn"
    url = f"wss://{host}/v2/iat"
    gmt = format_date_time(time.time())

    raw = f"host: {host}\ndate: {gmt}\nGET /v2/iat HTTP/1.1"
    sig = hmac.new(
        IFLYTEK_API_SECRET.encode(),
        raw.encode(),
        hashlib.sha256,
    ).digest()
    sig_b64 = base64.b64encode(sig).decode()

    auth_origin = (
        f'api_key="{IFLYTEK_API_KEY}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{sig_b64}"'
    )
    auth = base64.b64encode(auth_origin.encode()).decode()
    params = urlencode({"authorization": auth, "date": gmt, "host": host})
    return f"{url}?{params}"


# ============================================================
# AI 问答客户端
# ============================================================

class AIClient:
    """LLM 问答客户端，支持多轮对话"""

    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.history: List[Dict[str, str]] = []
        self._init_client()

    def _init_client(self):
        if not LLM_API_KEY:
            print_err("未配置 API Key，请设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
            return
        kwargs = {"api_key": LLM_API_KEY}
        if LLM_API_BASE:
            kwargs["base_url"] = LLM_API_BASE
        try:
            self.client = OpenAI(**kwargs)
            print_ok(f"AI 客户端已连接 -> {LLM_PROVIDER}/{LLM_MODEL}")
        except Exception as e:
            print_err(f"创建 AI 客户端失败: {e}")

    def ask(self, question: str) -> str:
        return self._ask_sync(question)

    def ask_stream(self, question: str) -> str:
        """流式输出: 逐 token 打印, 用户不用等"""
        return self._ask_sync(question, stream=True)

    def _ask_sync(self, question: str, stream: bool = False) -> str:
        if self.client is None:
            return "[错误] AI 客户端未配置，请在 .env 中设置 API Key"

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        max_history_msgs = MAX_HISTORY * 2
        messages.extend(self.history[-max_history_msgs:])
        messages.append({"role": "user", "content": question})

        try:
            resp = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
                stream=stream,
            )
            if stream:
                answer = ""
                print(_c(Colors.GREEN, "  "))
                for chunk in resp:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        print(delta.content, end="", flush=True)
                        answer += delta.content
                print()
                answer = answer.strip()
            else:
                answer = resp.choices[0].message.content or ""
                answer = answer.strip()
        except Exception as e:
            answer = f"[AI 出错] {str(e)}"
            print_err(f"API 调用失败: {e}")
            return answer

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer

    def clear_history(self):
        self.history.clear()
        print_info("对话历史已清除")


# ============================================================
# TTS 引擎 (可选)
# ============================================================

class TTSEngine:
    def __init__(self):
        self._engine = None
        if ENABLE_TTS and _HAS_TTS:
            try:
                self._engine = pyttsx3.init()
                self._engine.setProperty("rate", 190)
                self._engine.setProperty("volume", 0.9)
            except Exception as e:
                print_warn(f"TTS 初始化失败: {e}")

    def speak(self, text: str):
        if self._engine is None:
            return
        try:
            self._engine.stop()
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass

    def stop(self):
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


# ============================================================
# 键盘快捷键 (可选)
# ============================================================

class KeyboardWatcher(threading.Thread):
    """监听全局热键的守护线程 (需要 keyboard 库)"""

    def __init__(self, on_trigger=None, on_clear=None, on_quit=None):
        super().__init__(daemon=True)
        self._running = False
        self._on_trigger = on_trigger
        self._on_clear = on_clear
        self._on_quit = on_quit

    def run(self):
        if not _HAS_KEYBOARD:
            print_warn("keyboard 库未安装，热键不可用。请运行: pip install keyboard")
            return
        try:
            keyboard.add_hotkey("space", self._safe_call(self._on_trigger), suppress=False)
            keyboard.add_hotkey("c", self._safe_call(self._on_clear), suppress=False)
            keyboard.add_hotkey("q", self._safe_call(self._on_quit), suppress=False)
            keyboard.wait()  # 阻塞直到进程退出
        except Exception as e:
            print_warn(f"keyboard 注册失败 (可能需要管理员权限): {e}")

    def _safe_call(self, fn):
        def wrapper():
            try:
                if fn:
                    threading.Thread(target=fn, daemon=True).start()
            except Exception:
                pass
        return wrapper


# ============================================================
# 主程序
# ============================================================

class InterviewAssistant:
    """整合所有模块的主控制器"""

    def __init__(self):
        self.audio    = AudioStream()
        self.ai       = AIClient()
        self.stt      = STTEngine(self.ai.client)
        self.tts      = TTSEngine()
        self.running  = True

    def run(self):
        self._print_banner()
        self._list_audio_devices()

        try:
            self.audio.start()
        except Exception:
            return

        # 启动热键线程
        kb = KeyboardWatcher(
            on_trigger=self.manual_trigger,
            on_clear=self.ai.clear_history,
            on_quit=self.quit,
        )
        kb.start()

        print_ok("程序已启动! 自动监听中...")
        if _HAS_KEYBOARD:
            print("      [Space] 手动处理  |  [C] 清除历史  |  [Q] 退出")
        else:
            print("      [Ctrl+C] 退出程序")
        print()

        # 主循环
        try:
            if STT_PROVIDER == "xunfei":
                self._run_streaming_loop()
            else:
                self._run_batch_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self._shutdown()

    def _run_streaming_loop(self):
        """流式识别主循环: 持续累积所有语音, 直到 12 秒无人说话才触发 AI"""
        FINAL_SILENCE = 1.5

        all_parts: List[str] = []

        while self.running:
            # 等待第一个语音段
            print_info("等待面试官提问...")
            first = ""
            while self.running:
                try:
                    item = self.audio.stream_queue.get(timeout=0.1)
                except Empty:
                    continue
                if item is not AudioStream.STREAM_START:
                    continue
                print("\n")
                print_info("检测到语音，实时识别中...")
                first = self.stt.transcribe_streaming(self.audio.stream_queue, started=True)
                first = correct_keywords(first)
                if first and len(first.strip()) >= 2:
                    break
            if not first:
                continue

            all_parts = [first]
            silence_start = time.time()

            # 持续累积后续语音段
            while self.running:
                # 检查是否超时
                if time.time() - silence_start >= FINAL_SILENCE:
                    break

                try:
                    item = self.audio.stream_queue.get(timeout=0.1)
                except Empty:
                    continue
                if item is not AudioStream.STREAM_START:
                    continue

                # 有新语音
                print_info("检测到补充提问，实时识别中...")
                more = self.stt.transcribe_streaming(self.audio.stream_queue, started=True)
                more = correct_keywords(more)
                if more and len(more.strip()) >= 2:
                    all_parts.append(more)
                    print_info(f"已累积 {len(all_parts)} 段提问")
                    silence_start = time.time()  # 重置计时器

            question = "".join(all_parts)
            print_ok(f"共 {len(all_parts)} 段: {question}")

            quick = _match_quick_response(question)
            if quick:
                print_info("自我介绍 秒返!")
                print_q(question, quick)
                continue

            print_info("AI 思考中...")
            ai_start_t = time.time()
            AudioStream._lb_pause = True

            answer = self.ai.ask_stream(question)
            ai_time = time.time() - ai_start_t
            AudioStream._lb_pause = False

            print_q(question, stream_mode=True)
            print_info(f"AI 耗时: {ai_time:.1f}s")

            if ENABLE_TTS:
                threading.Thread(target=self.tts.speak, args=(answer,), daemon=True).start()

            all_parts = []

    def _run_batch_loop(self):
        """批量识别主循环: 兼容 openai / local 等"""
        while self.running:
            try:
                audio_data = self.audio.segment_queue.get(timeout=0.2)
                self._handle_segment(audio_data)
            except Empty:
                continue
            except Exception:
                traceback.print_exc()

    # ---------- 处理逻辑 ----------

    def _handle_segment(self, audio_bytes: bytes):
        duration = len(audio_bytes) / (SAMPLE_RATE * 2)
        print_info(f"检测到语音片段 ({duration:.1f}s)，正在识别...")

        question = self.stt.transcribe(audio_bytes)
        question = correct_keywords(question)
        if not question:
            print_warn("未识别到有效内容")
            return
        if len(question.strip()) < 2:
            return

        print_ok(f"识别: {question}")
        print_info("AI 回答生成中...")

        answer = self.ai.ask(question)
        print_q(question, answer)

        if ENABLE_TTS:
            threading.Thread(target=self.tts.speak, args=(answer,), daemon=True).start()

    def manual_trigger(self):
        """手动触发: 分析最近 5 秒音频"""
        audio_bytes = self.audio.get_recent(5.0)
        if len(audio_bytes) < SAMPLE_RATE * 2 * 0.5:  # < 0.5s
            print_warn("近 5 秒没有足够的音频数据")
            return
        print_ok("手动处理中...")
        self._handle_segment(audio_bytes)

    def quit(self):
        self.running = False

    # ---------- 辅助 ----------

    def _print_banner(self):
        print()
        print(_c(Colors.CYAN, "╔══════════════════════════════════════════════════╗"))
        print(_c(Colors.CYAN, "║       AI 面试助手  v1.0  —  Interview AI        ║"))
        print(_c(Colors.CYAN, "║    实时监听 · 语音识别 · 智能回答生成           ║"))
        print(_c(Colors.CYAN, "╚══════════════════════════════════════════════════╝"))
        stt_label = STT_PROVIDER
        if STT_PROVIDER == "local":
            stt_label += f"/{LOCAL_WHISPER_MODEL}"
        elif STT_PROVIDER == "xunfei":
            stt_label += "/iat"
        else:
            stt_label += f"/{STT_MODEL}"
        audio_src = "系统声音(loopback)" if AUDIO_MODE == "loopback" else "麦克风"
        print(f"  STT: {stt_label}  |  LLM: {LLM_PROVIDER}/{LLM_MODEL}  |  语言: {STT_LANGUAGE}")
        print(f"  音频: {audio_src}")
        print()

    def _list_audio_devices(self):
        try:
            devices = sd.query_devices()
            input_devices = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
            if input_devices:
                print_info(f"检测到 {len(input_devices)} 个输入设备:")
                for i, d in input_devices:
                    is_default = sd.default.device[0] == i
                    mark = " (默认)" if is_default else ""
                    print(f"      [{i}] {d['name']}{mark}")
                print()
        except Exception:
            pass

    def _shutdown(self):
        self.running = False
        self.tts.stop()
        self.audio.stop()
        print("\n再见~")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI 面试助手 - 实时监听面试音频并 AI 解答",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                          # 默认配置运行
  python main.py --stt local              # 使用本地 Whisper
  python main.py --language en            # 英文识别
  python main.py --tts                    # 开启语音播报
  python main.py --device 2               # 指定音频设备
        """,
    )
    parser.add_argument("--stt", choices=["openai", "local"], default=None, help="语音识别方式 (覆盖 .env)")
    parser.add_argument("--tts", action="store_true", default=None, help="启用文字转语音")
    parser.add_argument("--language", default=None, help="识别语言 (zh/en/... )")
    parser.add_argument("--device", type=int, default=None, help="音频输入设备编号")
    parser.add_argument("--energy", type=float, default=None, help="VAD 能量阈值 (0=自动, 建议 50-500)")
    parser.add_argument("--silence", type=float, default=None, help="静音阈值(秒)")
    parser.add_argument("--provider", default=None, help="LLM 提供商 (openai/deepseek/... )")
    parser.add_argument("--model", default=None, help="LLM 模型名")
    args = parser.parse_args()

    # 允许命令行参数覆盖全局配置
    global STT_PROVIDER, STT_LANGUAGE, ENABLE_TTS, DEVICE_INDEX, VAD_ENERGY_THRESHOLD
    global SILENCE_DURATION, LLM_PROVIDER, LLM_MODEL

    if args.stt:
        STT_PROVIDER = args.stt
    if args.language:
        STT_LANGUAGE = args.language
    if args.tts:
        ENABLE_TTS = True
    if args.device is not None:
        DEVICE_INDEX = args.device
    if args.energy is not None:
        VAD_ENERGY_THRESHOLD = args.energy
    if args.silence is not None:
        SILENCE_DURATION = args.silence
    if args.provider:
        LLM_PROVIDER = args.provider
    if args.model:
        LLM_MODEL = args.model

    # 检查 API Key
    if not LLM_API_KEY:
        print_err("未设置 API Key!")
        print_info("请创建 .env 文件并设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY")
        print_info("可以复制 .env.example 作为模板: copy .env.example .env")
        sys.exit(1)

    # 自动检测并加载简历
    global SYSTEM_PROMPT, _RESUME_TEXT
    resume = _find_and_read_resume()
    if resume:
        _RESUME_TEXT = resume
        SYSTEM_PROMPT = f"""{_SYSTEM_PROMPT_BASE}

## 面试者简历 (务必围绕以下真实经历回答)
{resume}

## 重要
回答时紧扣面试者简历中的技术栈和项目经验, 用简历中的真实项目举例, 不要编造经历。"""
        print_ok("已根据简历生成专属提示词")
    else:
        print_warn("未找到简历文件, 将使用通用模式 (建议将 .pdf 简历放在程序同目录下)")

    app = InterviewAssistant()
    app.run()


if __name__ == "__main__":
    main()
