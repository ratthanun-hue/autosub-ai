import os
import re

import math
import time
import json
import threading
import subprocess
import tempfile
import logging
from typing import Optional
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, JSONResponse
import urllib.request

# ML / Audio engines
import torch
import numpy as np
import soundfile as sf
import torchaudio
import torchaudio.functional as AF
from transformers import WhisperFeatureExtractor, WhisperModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download, snapshot_download
from pythainlp.tokenize import word_tokenize
from pythainlp.util import Trie
from pythainlp.corpus.common import thai_words
from faster_whisper import WhisperModel as FasterWhisperModel
from faster_whisper.vad import get_speech_timestamps, VadOptions

app = FastAPI(title="Dual-Engine Subtitle API (Typhoon Thai + Whisper Multilingual)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# SUPPRESS ROUTINE POLLING FROM ACCESS LOGS
# ==========================================
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for ep in ["/health", "/logs", "/gpu", "/touch"]:
            if ep in msg:
                return False
        return True

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())


# ==========================================
# AUTO-SLEEP & WATCHDOG CONFIGURATION
# ==========================================
IDLE_TIMEOUT_SECONDS = 14400  # 4 hours idle limit
INSTANCE_ID = "51322461"
try:
    if os.path.exists("/root/.vast_containerlabel"):
        with open("/root/.vast_containerlabel", "r") as _f:
            _cid = _f.read().strip()
            if _cid.isdigit():
                INSTANCE_ID = _cid
except Exception:
    pass
VAST_API_KEY_DEFAULT = "bb158182f28dba3c4d30c71fd31eca1149c65b308b7f59ead54c0a5c66332a5d"

state_lock = threading.Lock()
last_active_time = time.time()
active_jobs = 0
total_jobs_completed = 0
is_shutting_down = False

# Cache static GPU specs
GPU_NAME = "NVIDIA GeForce RTX 3090"
GPU_VRAM_GB = 24.0
try:
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5
    )
    if smi.returncode == 0 and smi.stdout.strip():
        parts = [p.strip() for p in smi.stdout.strip().split(",")]
        GPU_NAME = parts[0]
        if len(parts) > 1:
            GPU_VRAM_GB = round(float(parts[1]) / 1024, 2)
except Exception:
    pass

TRANSCRIBE_LOG_PATH = "/root/whisper-server/transcribe.log"
SERVER_LOG_PATH = "/root/whisper-server/server.log"

import contextvars
current_file_ctx = contextvars.ContextVar("current_file_ctx", default="")

def log_transcribe(msg: str, filename: str = None):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    file_tag = filename or current_file_ctx.get()
    prefix = f"[{file_tag}] " if file_tag else ""
    formatted = f"[{timestamp}] {prefix}{msg}\n"
    print(formatted, end="", flush=True)
    try:
        if os.path.exists(TRANSCRIBE_LOG_PATH) and os.path.getsize(TRANSCRIBE_LOG_PATH) > 10 * 1024 * 1024:
            for i in [2, 1]:
                old = f"{TRANSCRIBE_LOG_PATH}.{i}"
                nxt = f"{TRANSCRIBE_LOG_PATH}.{i+1}"
                if os.path.exists(old):
                    try:
                        os.replace(old, nxt)
                    except Exception:
                        pass
            try:
                os.replace(TRANSCRIBE_LOG_PATH, f"{TRANSCRIBE_LOG_PATH}.1")
            except Exception:
                pass

        with open(TRANSCRIBE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(formatted)
    except Exception as ex:
        print(f"Log write error: {ex}", flush=True)

def update_activity():
    global last_active_time
    with state_lock:
        last_active_time = time.time()

def trigger_auto_sleep():
    global is_shutting_down
    with state_lock:
        if is_shutting_down or active_jobs > 0:
            return
        is_shutting_down = True

    msg = f"[WARNING] Server idle เกิน {IDLE_TIMEOUT_SECONDS}s ระบบกำลังสั่ง Auto-Sleep ดับเครื่อง {INSTANCE_ID} เพื่อประหยัดค่าใช้จ่าย..."
    log_transcribe(msg)
    print(f"\n[Auto-Sleep] Server idle for >= {IDLE_TIMEOUT_SECONDS}s. Initiating Vast.ai stop instance {INSTANCE_ID}...\n", flush=True)

    try:
        cmd = ["/opt/instance-tools/bin/vastai", "stop", "instance", INSTANCE_ID]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if res.returncode == 0:
            return
    except Exception as e:
        print(f"[Auto-Sleep] vastai CLI failed: {e}", flush=True)

    try:
        key_candidates = ["/root/.config/vastai/vast_api_key", "/root/.vast_api_key"]
        api_key = VAST_API_KEY_DEFAULT
        for kp in key_candidates:
            if os.path.exists(kp):
                with open(kp, "r") as f:
                    content = f.read().strip()
                if content:
                    api_key = content
                    break

        if api_key:
            import urllib.request
            url = f"https://console.vast.ai/api/v0/instances/{INSTANCE_ID}/"
            payload = json.dumps({"state": "stopped"}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                method="PUT"
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                print(f"[Auto-Sleep] REST API fallback result: {resp.status} {resp.read().decode('utf-8')}", flush=True)
    except Exception as e2:
        print(f"[Auto-Sleep] REST API fallback failed: {e2}", flush=True)

def watchdog_loop():
    global last_active_time, active_jobs, is_shutting_down
    while True:
        time.sleep(10)
        with state_lock:
            idle_elapsed = time.time() - last_active_time
            busy = active_jobs > 0
            shutting_down = is_shutting_down

        if shutting_down:
            break

        if idle_elapsed >= IDLE_TIMEOUT_SECONDS and not busy:
            trigger_auto_sleep()
            break

watchdog_thread = threading.Thread(target=watchdog_loop, daemon=True)
watchdog_thread.start()

# ==========================================
# 1. ENGINE INITIALIZATION: MULTILINGUAL TURBO
# ==========================================
print("Initializing Multilingual Engine: Whisper large-v3-turbo (CUDA FP16)...")
turbo_model = FasterWhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
print("Multilingual Engine (Chinese/Korean/English/Japanese) Ready!")

# ==========================================
# 2. ENGINE INITIALIZATION: TYPHOON THAI CTC
# ==========================================
print("Initializing Thai Engine: Typhoon-Whisper-large-v3-ctc (CUDA FP16)...")
device = "cuda"

print("Fetching Typhoon CTC Head and Encoder from Hugging Face...")
try:
    ctc_dir = snapshot_download(repo_id="typhoon-ai/typhoon-whisper-large-v3-ctc")
except Exception as e:
    print("snapshot_download ctc error, falling back to default path:", e)
    ctc_dir = "/root/.cache/huggingface/hub/models--typhoon-ai--typhoon-whisper-large-v3-ctc/snapshots/4b7b7b836fb98d58972a1a44cf2b8ced56564345"

try:
    enc_dir = snapshot_download(repo_id="typhoon-ai/typhoon-whisper-large-v3")
except Exception as e:
    print("snapshot_download enc error, falling back to default path:", e)
    enc_dir = "/root/.cache/huggingface/hub/models--typhoon-ai--typhoon-whisper-large-v3/snapshots/748e8a418697d5920e62bced10590072dce90da5"


ctc_cfg = json.loads(open(f"{ctc_dir}/config.json").read())
ctc_symbols = json.loads(open(f"{ctc_dir}/ctc_vocab.json").read())["symbols"]
ctc_sd = load_file(f"{ctc_dir}/head.safetensors")

import torch.nn as nn
d_in, d = ctc_sd["proj.weight"].shape[1], ctc_sd["proj.weight"].shape[0]
ffn = ctc_sd["layers.layers.0.linear1.weight"].shape[0]
n_layers = 1 + max(int(k.split(".")[2]) for k in ctc_sd if k.startswith("layers.layers."))

class CTCHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.act = nn.GELU()
        layer = nn.TransformerEncoderLayer(d, int(ctc_cfg["n_heads"]), ffn, activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, len(ctc_symbols))

    def forward(self, h):
        return self.out(self.ln(self.layers(self.act(self.proj(h)))))

typhoon_head = CTCHead()
typhoon_head.load_state_dict(ctc_sd, strict=True)
typhoon_head = typhoon_head.to(device).eval()

typhoon_fe = WhisperFeatureExtractor.from_pretrained(enc_dir)
typhoon_enc = WhisperModel.from_pretrained(
    enc_dir,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).encoder.to(device).eval()



ctc_sym2id = {s: i for i, s in enumerate(ctc_symbols)}
print("Thai Engine (Typhoon + CTC + PyThaiNLP) Ready!")
print(f"Dual-Engine Subtitle Server Ready on RTX 3090 ({GPU_NAME})! 4-hour Auto-Sleep active.")

# ==========================================
# VOCAL ISOLATION ENGINE (HDemucs v4 MusDB+)
# ==========================================
print("Initializing Vocal Isolation Engine: HDemucs (MusDB+)...")
demucs_model = None
try:
    torch.backends.cudnn.enabled = False
    demucs_bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
    demucs_model = demucs_bundle.get_model().to(device).eval()
    print("Vocal Isolation Engine Ready on GPU!")
except Exception as _demucs_err:
    print(f"Demucs GPU init notice ({_demucs_err}) - Demucs will be optional.")
    try:
        demucs_bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
        demucs_model = demucs_bundle.get_model().to("cpu").eval()
        print("Vocal Isolation Engine Ready on CPU!")
    except Exception:
        demucs_model = None

def isolate_vocals_from_audio(wav: np.ndarray, orig_sr: int, chunk_sec: int = 60) -> torch.Tensor:
    """
    Separates human speech from background music (BGM/OST) using HDemucs v4.
    Returns clean 16000Hz mono float32 tensor of pure vocal speech.
    """
    if wav.ndim == 1:
        wav_stereo = np.stack([wav, wav], axis=0)
    else:
        wav_stereo = wav.T if wav.shape[0] > wav.shape[1] else wav

    t_stereo = torch.as_tensor(wav_stereo, dtype=torch.float32)
    if orig_sr != 44100:
        t_stereo = AF.resample(t_stereo, orig_sr, 44100)

    sr = 44100
    chunk_samples = chunk_sec * sr
    total_samples = t_stereo.shape[-1]
    vocals_chunks = []

    for i in range(0, total_samples, chunk_samples):
        chunk = t_stereo[:, i:i + chunk_samples]
        if chunk.shape[-1] < sr * 2:
            vocals_chunks.append(chunk.mean(dim=0))
            continue
        with torch.no_grad():
            inp = chunk.unsqueeze(0).to(device)
            sources = demucs_model(inp)
            voc = sources[0, 3].mean(dim=0).cpu()  # Index 3 is Vocals
            vocals_chunks.append(voc)

    vocal_44k = torch.cat(vocals_chunks, dim=-1)
    return AF.resample(vocal_44k, 44100, 16000)

# ==========================================
# DIALOGUE & SUBTITLE SEGMENTATION
# ==========================================
ENDING_PARTICLES = {
    "ครับ", "ค่ะ", "คะ", "จ้ะ", "จ้า", "จ๊ะ", "นะ", "ไหม", "มั้ย", "เหรอ", "หรอ", "หรือเปล่า",
    "อ่ะ", "วะ", "เว้ย", "โว้ย", "สิ", "ล่ะ", "ละ",
    "เออ", "สัด", "สัส", "สัตว์", "เหี้ย", "เชี่ย",
    "นะเนี่ย", "เนี่ยนะ", "ใช่ไหม", "ใช่มั้ย", "เนี่ย", "หรอก", "ไง"
}

# ==========================================
# THAI SLANG VOCABULARY & CUSTOM TRIE
# ==========================================
_SLANG_WORDS_PATH = os.path.join(os.path.dirname(__file__), "thai_slang_words.json")
SLANG_WORDS = []
if os.path.exists(_SLANG_WORDS_PATH):
    try:
        with open(_SLANG_WORDS_PATH, "r", encoding="utf-8") as _f:
            SLANG_WORDS = json.load(_f)
    except Exception as _e:
        logging.warning(f"Could not load slang words from file: {_e}")

if not SLANG_WORDS:
    SLANG_WORDS = [
        "ตัวแม่", "ตัวมัม", "ตัวมารดา", "จึ้ง", "ฉ่ำ", "โฮ่ง", "คุณน้า", "ฟีล", "เกินต้าน",
        "ปัง", "ยืนหนึ่ง", "เต็มคาราเบล", "ที่สุด", "งานละเอียด", "ฉลาม", "จริต", "ตาแตก", "มงลง",
        "ชี", "ฮี", "ช็อตฟีล", "นอยด์", "บูด", "ขิต", "ขุด", "แกง", "โป๊ะ", "ตุ๊บ", "เท", "บ้ง",
        "หน้าแหก", "มโน", "มองบน", "ลำไย", "สภาพ", "จม", "แห้ว", "นก", "จึ้งมาก", "จึ้งใจ",
        "ฉ่ำมาก", "สวยฉ่ำ", "เลิศฉ่ำ", "นอยด์อ่า", "นอยด์มาก", "นอยด์แดก", "บ้งมาก", "บ้งสุด",
        "ขิตหมู่", "สู่ขิต", "แกงหม้อใหญ่", "โดนแกง", "โป๊ะแตก", "จับโป๊ะ", "ปังมาก", "ปังปุริเย่",
        "ปังไม่ไหว", "ดีย์", "งานดี", "เกินต้านทาน", "ช็อตฟีลแรง", "โดนช็อตฟีล", "ตัวพ่อ", "ตัวแด๊ด",
        "ตัวบิดา", "สุดจัด", "ปลัดบอก", "ต๊าช", "อ่อม", "นอยด์น้า", "ตัวตึง", "ตึงเปรี้ยะ", "แซ่บเวอร์",
        "แซ่บนัว", "เดินสับ", "สับขาหลอก", "ตัวเต็ง", "เต็งหนึ่ง", "ม้ามืด", "วงวาร", "ไอต้าว", "ต้าวอ้วน",
        "นุ่มฟู", "ใจฟู", "ละมุนนี", "ตะมุตะมิ", "บิดงาน", "สายมู", "มูฉ่ำ", "แรร์ไอเทม", "ของมันต้องมี",
        "ป้ายยา", "โดนป้ายยา", "อวยยศ", "โดนสปอยล์", "ทริปล่ม", "ตัวบั๊ก", "ติงต๊อง", "เด๋อด๋า",
        "ทรงอย่างแบด", "แซดอย่างบ่อย", "แจกวาร์ป", "แฉยับ", "อึ้งกิมกี่", "สลบเหมือด", "เก็ทป่ะ",
        "จริงดิ", "ว่าซั่น", "ชัวร์ปึ้ก", "ชัวร์ป๊าบ", "เริ่มเลอ", "จัดไปอย่าให้เสีย", "สายเปย์",
        "แฮงเอาท์", "ปวดตับ", "ดราม่าควีน", "ฟินเฟ่อร์", "ฟินกระจาย", "หัวร้อน", "ขัดใจสิ่งนี้",
        "เหม็นขี้หน้า", "ประสาทแดก", "ประสาทจะกิน", "กวนตีน", "กวนบาทา", "กวนโอ๊ย", "บ้าบอคอแตก",
        "ฉิบหายวายวอด", "เวรซ้ำกรรมซ้อน", "ปลิ้นปล้อนกะล่อนทอง", "หน้าส้นตีน", "ส้มตำ", "หมูกระทะ",
        "กะเพราไข่ดาว", "เบิร์นเอาท์", "เดดไลน์", "งานงอก", "ขายฝัน", "ล่มปากอ่าว", "ตัวแบก",
        "พ่อไมโครเวฟ", "พี่น้องโซน", "เฟรนด์โซน", "สายซัพ", "พ่อบ้านใจกล้า", "คุมโหด",
        "กู", "มึง", "ไอ้", "วะ", "เว้ย", "โถฉี่", "ห้องน้ำ", "คุณยาดา", "เพชรแท้", "เพชรประกาย", "โถเตอะ"
    ]

try:
    CUSTOM_TRIE = Trie(set(thai_words()).union(set(SLANG_WORDS)))
except Exception as _e:
    logging.warning(f"Could not build Trie with thai_words: {_e}")
    CUSTOM_TRIE = Trie(set(SLANG_WORDS))

def thai_tokenize(text: str):
    if not text:
        return []
    try:
        return [w for w in word_tokenize(text, custom_dict=CUSTOM_TRIE, engine="newmm") if w.strip()]
    except Exception:
        return [w for w in word_tokenize(text, engine="newmm") if w.strip()]

# ==========================================
# THAI AUTOMATIC CORRECTIONS & PHONETIC MAP
# ==========================================
_CORRECTIONS_PATH = os.path.join(os.path.dirname(__file__), "thai_corrections.json")
THAI_CORRECTIONS = {}
if os.path.exists(_CORRECTIONS_PATH):
    try:
        with open(_CORRECTIONS_PATH, "r", encoding="utf-8") as _f:
            THAI_CORRECTIONS = json.load(_f)
    except Exception as _e:
        logging.warning(f"Could not load corrections from file: {_e}")

if not THAI_CORRECTIONS:
    THAI_CORRECTIONS = {
        r"ประจุย": "กระจุย",
        r"แนะนา": "แนะนำ",
        r"เมื่อกันสวย": "ไม่งั้นซวย",
        r"เมื่อกั้นสวย": "ไม่งั้นซวย",
        r"สำหิจารณ์": "สามีจ๋า",
        r"เมียจาร์": "เมียจ๋า",
        r"รายการต่อ[ปบ]ีนี้": "รายการต่อไปนี้",
        r"รายการต่อเป็น[นี้อีย]+": "รายการต่อไปนี้",
        r"ด้วยการต่อเป็นหน้า": "รายการต่อไปนี้",
        r"เป็นโดยการทั่วป่า": "เป็นรายการทั่วไป",
        r"สามารถ?รับ[ทช]ันได้ทุก[ว่ายวัน]+": "สามารถรับชมได้ทุกวัย",
        r"สามารถ?รับ[ทช]มได้ทุก[ว่ายวัน]+": "สามารถรับชมได้ทุกวัย",
        r"สามารับชมได้ทุกวัย": "สามารถรับชมได้ทุกวัย",
        r"สามารับทมได้ทุกวัย": "สามารถรับชมได้ทุกวัย",
        r"ทุกว่าย": "ทุกวัย",
        r"วิจ[รณรร]+[ยณาน]+": "วิจารณญาณ",
        r"วิทยาลนยาน": "วิจารณญาณ",
        r"คุณโกษ": "คุณโกรธ",
        r"คุณผ่วย": "คุณป่วย",
        r"หัว่คุณป่วย": "หวังว่าคุณป่วย",
        r"ยังไงบ้าน": "ยังไงบ้าง",
        r"โทษฉี่?": "โถฉี่",
        r"โถ่ฉี่?": "โถฉี่",
        r"โทษเติด": "โถเตอะ",
        r"โถเติด": "โถเตอะ",
        r"โตเต": "โถเตอะ",
        r"เข้าน้ำ": "ห้องน้ำ",
        r"เข้าห้น้ำ": "เข้าห้องน้ำ",
        r"กูล้ำเกล้ำลงว่ะ": "กูแล้วมีอารมณ์ว่ะ",
        r"กูล้มเก้าลงว่ะ": "กูแล้วมีอารมณ์ว่ะ",
        r"กูล้ำเกลงลงว่ะ": "กูแล้วมีอารมณ์ว่ะ",
        r"คุณยาดาย": "คุณยาดา",
        r"คุย่าดา": "คุณยาดา",
        r"เผ็ดแท้": "เพชรแท้",
        r"เผ็ดประกาย": "เพชรประกาย",
        r"ถักสะ": "ทักษะ",
        r"ลิกขะสิด": "ลิขสิทธิ์",
        r"พาตสปอด": "พาสปอร์ต",
        r"คอนเส็บ": "คอนเซ็ปต์",
        r"เอดกะสาน": "เอกสาร",
        r"พาดเวิด": "พาสเวิร์ด",
        r"สะหมาดโฟน": "สมาร์ทโฟน",
        r"ออนไล": "ออนไลน์",
        r"ออฟฟิด": "ออฟฟิศ",
        r"แบดเตอรี่": "แบตเตอรี่",
        r"คีบอด": "คีย์บอร์ด",
        r"จอพาบ": "จอภาพ",
        r"สะแกนเนอร์": "สแกนเนอร์",
        r"แท็บแล็ต": "แท็บเล็ต",
        r"ไอแปด": "ไอแพด",
        r"คอมพิวเต้อ": "คอมพิวเตอร์",
        r"ติดตัง": "ติดตั้ง",
        r"รีสต๊าด": "รีสตาร์ท",
        r"รีเซด": "รีเซ็ต",
        r"เออเร่อ": "เออร์เรอร์",
        r"เว็บบิน่า": "เว็บบินาร์",
        r"กะเพา": "กะเพรา",
        r"อนุญาติ": "อนุญาต"
    }

def correct_thai_transcription(text: str) -> str:
    if not text:
        return text
    global THAI_CORRECTIONS
    try:
        if os.path.exists(_CORRECTIONS_PATH):
            mtime = os.path.getmtime(_CORRECTIONS_PATH)
            if getattr(correct_thai_transcription, "_last_mtime", 0) < mtime:
                with open(_CORRECTIONS_PATH, "r", encoding="utf-8") as _f:
                    THAI_CORRECTIONS = json.load(_f)
                correct_thai_transcription._last_mtime = mtime
    except Exception:
        pass
    for pattern, repl in THAI_CORRECTIONS.items():
        text = re.sub(pattern, repl, text)
    return text

DEFAULT_DRAMA_PROMPT = (
    "รายการต่อไปนี้เป็นรายการทั่วไป สามารถรับชมได้ทุกวัย เหมาะสำหรับผู้ชมที่มีอายุ 13 ปีขึ้นไป "
    "อาจมีภาพ เสียง หรือเนื้อหาที่ต้องใช้วิจารณญาณในการรับชม บทสนทนาละครและซีรีส์ไทย "
    "ภาษาพูดสแลง: ตัวแม่, ตัวมัม, จึ้ง, ฉ่ำ, นอยด์, บูด, โป๊ะ, บ้ง, ช็อตฟีล, แกง, มโน, สภาพ, "
    "เต็มคาราเบล, ปังมาก, ปังปุริเย่, ดีย์, ตัวตึง, แซ่บนัว, สายมู, ป้ายยา, ทรงอย่างแบด, แฉยับ, "
    "หัวร้อน, ประสาทแดก, กวนตีน, ห้องน้ำ, โถฉี่, มีอารมณ์, โกรธ, ทำไมวะ, อะไรวะ, กู, มึง, ไอ้, ใคร, ไปไหน, "
    "ไม่เป็นไร, เข้าใจ, ปัญหา, รักษา, ป่วย, หมอ, โรงพยาบาล, นะคะ, ครับ, จ้ะ, วะ, เว้ย"
)


def form_dialogue_segments(all_words, max_chars_per_cue: int = 70, max_pause_sec: float = 0.22, min_cue_dur: float = 0.8, time_offset: float = -0.55, max_cue_dur: float = 5.0):
    segments = []
    curr = []
    seg_id = 0

    # Shift words earlier by time_offset to eliminate the ~1s audio-to-subtitle lag
    shifted_words = []
    for w in all_words:
        s = max(0.0, round(w["start"] + time_offset, 3))
        e = max(s + 0.05, round(w["end"] + time_offset, 3))
        shifted_words.append({
            "word": w["word"],
            "start": s,
            "end": e,
            "conf": w.get("conf", 0.9)
        })

    for w in shifted_words:
        if not curr:
            curr.append(w)
            continue
        pause = w["start"] - curr[-1]["end"]
        cur_txt = "".join(x["word"] for x in curr)
        is_ending_particle = curr[-1]["word"] in ENDING_PARTICLES
        should_split = (
            pause > max_pause_sec or
            len(cur_txt) + len(w["word"]) > max_chars_per_cue or
            (is_ending_particle and pause >= 0.08)
        )
        if should_split:
            c_start = round(curr[0]["start"], 3)
            raw_end = curr[-1]["end"]
            next_start = w["start"]
            c_end = round(min(max(c_start + 0.2, next_start - 0.04), max(raw_end, c_start + min_cue_dur)), 3)
            # Cap max display duration to prevent subtitle staying on screen too long
            final_end = max(round(c_start + 0.3, 3), c_end)
            if final_end - c_start > max_cue_dur:
                final_end = round(c_start + max_cue_dur, 3)
            clean_txt = correct_thai_transcription(cur_txt)
            segments.append({
                "id": seg_id,
                "start": c_start,
                "end": final_end,
                "text": clean_txt
            })
            seg_id += 1
            curr = [w]
        else:
            curr.append(w)

    if curr:
        c_start = round(curr[0]["start"], 3)
        c_end = round(max(curr[-1]["end"], c_start + min_cue_dur), 3)
        # Cap max display duration
        if c_end - c_start > max_cue_dur:
            c_end = round(c_start + max_cue_dur, 3)
        clean_txt = correct_thai_transcription("".join(x["word"] for x in curr))
        segments.append({
            "id": seg_id,
            "start": c_start,
            "end": c_end,
            "text": clean_txt
        })

    return segments

def format_timestamp(seconds: float, vtt: bool = False) -> str:
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds % 3600) / 60)
    secs = math.floor(seconds % 60)
    msecs = math.floor((seconds - math.floor(seconds)) * 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{msecs:03d}"

def transcribe_with_typhoon(audio_path: str, max_chars_per_cue: int = 70, max_pause_sec: float = 0.22, min_cue_dur: float = 0.9, isolate_vocals: bool = False):
    """
    End-to-End Thai Transcription & Alignment via Typhoon CTC:
    1. Vocal Isolation (HDemucs) -> optional BGM removal
    2. Dynamic Range / RMS Normalization -> boosts whisper & dialogue
    3. VAD slicing (Silero VAD, sensitive 0.45) -> prevents dropping soft speech
    4. Whisper Encoder + CTC Head -> SOTA Thai phoneme decoding
    5. CTC Forced Alignment -> exact 20ms timestamps
    6. PyThaiNLP word tokenization & grouping -> beautiful subtitle segments
    """
    wav, sr = sf.read(audio_path, dtype="float32")
    if isolate_vocals:
        try:
            x = isolate_vocals_from_audio(wav, sr)
        except Exception as e:
            print(f"[WARN] Vocal isolation exception: {e}, using direct audio")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            x = torch.as_tensor(wav, dtype=torch.float32)
            if sr != 16000:
                x = AF.resample(x, sr, 16000)
    else:
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        x = torch.as_tensor(wav, dtype=torch.float32)
        if sr != 16000:
            x = AF.resample(x, sr, 16000)

    # Audio Pre-processing: Loudness / RMS Normalization (Option 1)
    rms = torch.sqrt(torch.mean(x ** 2))
    if rms > 1e-5:
        target_rms = 0.08
        gain = min(target_rms / rms, 5.0)  # Capped at 5x gain to prevent boosting pure silence
        x = x * gain
    x = torch.clamp(x, -0.98, 0.98)

    total_duration = len(x) / 16000.0

    # 1. Voice Activity Detection (threshold 0.45 to catch whispers and soft speech)
    vad_opts = VadOptions(threshold=0.45, min_silence_duration_ms=250, speech_pad_ms=150)
    vad_segments = get_speech_timestamps(x, vad_options=vad_opts)
    all_words = []

    # Fallback if VAD returns nothing (e.g. continuous speech or low volume)
    if not vad_segments:
        vad_segments = [{"start": 0, "end": len(x)}]

    for seg in vad_segments:
        start_samp, end_samp = seg["start"], seg["end"]
        chunk = x[start_samp:end_samp]
        if len(chunk) / 16000.0 < 0.2:
            continue

        # Slices > 29s to fit Whisper receptive field
        max_samples = 29 * 16000
        for i in range(0, len(chunk), max_samples):
            sub_wav = chunk[i:i + max_samples]
            offset_s = (start_samp + i) / 16000.0

            fb = typhoon_fe(sub_wav.numpy(), sampling_rate=16000, return_attention_mask=True, return_tensors="pt")
            mel_len = int(fb.attention_mask.sum())
            with torch.no_grad():
                h = typhoon_enc(input_features=fb.input_features.to(device, typhoon_enc.dtype)).last_hidden_state[:, : (mel_len - 1) // 2 + 1]
                logits = typhoon_head(h.float())
                ids = logits.argmax(-1)[0].tolist()

            out, prev = [], 0
            for char_id in ids:
                if char_id != 0 and char_id != prev:
                    out.append(ctc_symbols[char_id])
                prev = char_id
            txt = "".join(out)
            if not txt.strip():
                continue

            target_ids, chars = [], []
            for ch in txt:
                key = " " if ch.isspace() else ch
                if key in ctc_sym2id and key != " ":
                    target_ids.append(ctc_sym2id[key])
                    chars.append(ch)

            if not target_ids:
                continue

            with torch.no_grad():
                logp = torch.log_softmax(logits, dim=-1).cpu()

            labels, scores = AF.forced_align(logp, torch.tensor([target_ids], dtype=torch.int32), blank=0)
            spans, lab = [], labels[0].tolist()
            t, k = 0, 0
            while t < len(lab):
                if lab[t] == 0:
                    t += 1
                    continue
                t0 = t
                while t + 1 < len(lab) and lab[t + 1] == lab[t0]:
                    t += 1
                s = max(0.0, t0 * 0.02) + offset_s
                e = max(s, (t + 1) * 0.02) + offset_s
                spans.append({"char": chars[k], "start": round(s, 3), "end": round(e, 3), "conf": round(float(scores[0][t0:t + 1].mean()), 3)})
                k += 1
                t += 1

            # Group into natural words with PyThaiNLP
            raw_words = thai_tokenize(txt.replace(" ", ""))
            k = 0
            for w in raw_words:
                grp = spans[k:k + len(w)]
                if not grp:
                    break
                w_start = grp[0]["start"]
                w_end = grp[-1]["end"]
                # Cap trailing vowel duration so background music does not stretch the word into next dialogue
                max_w_dur = max(0.50, len(w) * 0.20)
                if w_end - w_start > max_w_dur:
                    w_end = round(w_start + max_w_dur, 3)

                all_words.append({
                    "word": w,
                    "start": w_start,
                    "end": w_end,
                    "conf": round(sum(g["conf"] for g in grp) / len(grp), 3)
                })
                k += len(w)

    segments = form_dialogue_segments(all_words, max_chars_per_cue, max_pause_sec, min_cue_dur)
    full_text = " ".join(s["text"] for s in segments)
    return segments, full_text, total_duration

def transcribe_with_turbo_thai(
    audio_path: str,
    initial_prompt: Optional[str] = None,
    max_chars_per_cue: int = 70,
    max_pause_sec: float = 0.22,
    min_cue_dur: float = 0.8,
    temperature: float = 0.0,
    time_offset: float = -0.55,
):
    """
    End-to-End Thai Transcription & Alignment via Whisper large-v3-turbo:
    1. Dynamic RMS Normalization -> safely boosts soft speech & whispers
    2. Large-v3-turbo Transformer Decoder -> context-aware, zero dropped sentences
    3. Word-level Timestamps -> accurate alignment
    4. PyThaiNLP Tokenization -> natural Thai word boundaries
    5. Dialogue Segmentation -> ENDING_PARTICLES split, pause >= 0.08s, max 70 chars
    """
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        t_wav = torch.as_tensor(wav, dtype=torch.float32)
        wav = AF.resample(t_wav, sr, 16000).numpy()

    # Dynamic Loudness / RMS Normalization
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms > 1e-5:
        target_rms = 0.08
        gain = min(target_rms / rms, 5.0)
        wav = np.clip(wav * gain, -0.98, 0.98)

    total_duration = round(len(wav) / 16000.0, 2)

    if not initial_prompt or not initial_prompt.strip():
        initial_prompt = DEFAULT_DRAMA_PROMPT
    else:
        initial_prompt = initial_prompt.strip() + " " + DEFAULT_DRAMA_PROMPT

    segments_gen, info = turbo_model.transcribe(
        wav,
        language="th",
        temperature=temperature,
        beam_size=5,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        repetition_penalty=1.15,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=400, threshold=0.25)
    )

    all_words = []
    for s in segments_gen:
        char_spans = []
        if s.words:
            for w in s.words:
                clean = w.word.strip()
                if not clean:
                    continue
                dur = max(0.01, w.end - w.start)
                ch_dur = dur / max(1, len(clean))
                c_s = w.start
                for ch in clean:
                    char_spans.append({
                        "char": ch,
                        "start": round(c_s, 3),
                        "end": round(c_s + ch_dur, 3),
                        "conf": round(w.probability, 3)
                    })
                    c_s += ch_dur
        else:
            clean = s.text.strip()
            if not clean:
                continue
            dur = max(0.01, s.end - s.start)
            ch_dur = dur / max(1, len(clean))
            c_s = s.start
            for ch in clean:
                char_spans.append({
                    "char": ch,
                    "start": round(c_s, 3),
                    "end": round(c_s + ch_dur, 3),
                    "conf": 0.90
                })
                c_s += ch_dur

        if not char_spans:
            continue

        full_seg_text = "".join(c["char"] for c in char_spans)
        th_words = thai_tokenize(full_seg_text)
        idx = 0
        for tw in th_words:
            grp = char_spans[idx:idx + len(tw)]
            if not grp:
                break
            w_start = grp[0]["start"]
            w_end = grp[-1]["end"]
            # Cap trailing vowel duration so background music does not stretch the word
            max_w_dur = max(0.50, len(tw) * 0.20)
            if w_end - w_start > max_w_dur:
                w_end = round(w_start + max_w_dur, 3)
            all_words.append({
                "word": tw,
                "start": w_start,
                "end": w_end,
                "conf": round(sum(g["conf"] for g in grp) / len(grp), 3)
            })
            idx += len(tw)

    segments = form_dialogue_segments(all_words, max_chars_per_cue, max_pause_sec, min_cue_dur, time_offset=time_offset)
    full_text = " ".join(s["text"] for s in segments)
    return segments, full_text, total_duration

def transcribe_hybrid_thai(
    audio_path: str,
    initial_prompt: Optional[str] = None,
    max_chars_per_cue: int = 70,
    max_pause_sec: float = 0.22,
    min_cue_dur: float = 0.8,
    temperature: float = 0.0,
    time_offset: float = -0.55,
):
    """
    Hybrid Pipeline: Turbo for transcription + Typhoon CTC for 20ms alignment.
    Step 1: Whisper large-v3-turbo -> full text (100% coverage, no dropped words)
    Step 2: Typhoon CTC Forced Alignment -> 20ms character-level timestamps
    Step 3: PyThaiNLP word grouping -> natural Thai word boundaries
    Step 4: Dialogue segmentation with ENDING_PARTICLES
    Fallback: If CTC alignment fails for a segment, use Turbo word timestamps instead
    """
    wav, sr = sf.read(audio_path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        t_wav = torch.as_tensor(wav, dtype=torch.float32)
        wav = AF.resample(t_wav, sr, 16000).numpy()

    # Dynamic Loudness / RMS Normalization
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms > 1e-5:
        target_rms = 0.08
        gain = min(target_rms / rms, 5.0)
        wav = np.clip(wav * gain, -0.98, 0.98)

    total_duration = round(len(wav) / 16000.0, 2)

    if not initial_prompt or not initial_prompt.strip():
        initial_prompt = DEFAULT_DRAMA_PROMPT
    else:
        initial_prompt = initial_prompt.strip() + " " + DEFAULT_DRAMA_PROMPT

    # ===== STEP 1: Turbo transcription (full coverage) =====
    segments_gen, info = turbo_model.transcribe(
        wav,
        language="th",
        temperature=temperature,
        beam_size=5,
        word_timestamps=True,
        initial_prompt=initial_prompt,
        condition_on_previous_text=True,
        repetition_penalty=1.15,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=400, speech_pad_ms=400, threshold=0.25)
    )

    turbo_segments = list(segments_gen)
    log_transcribe(f"[HYBRID] Step 1 done: Turbo produced {len(turbo_segments)} raw segments")

    # Convert wav to torch tensor for CTC alignment
    x_16k = torch.as_tensor(wav, dtype=torch.float32)

    all_words = []
    ctc_aligned_count = 0
    turbo_fallback_count = 0

    for seg in turbo_segments:
        seg_text = seg.text.strip()
        if not seg_text:
            continue

        # ===== STEP 2: Try Typhoon CTC Forced Alignment for this segment =====
        ctc_success = False
        try:
            # Extract audio for this segment with 0.5s padding on each side
            pad = 0.5
            start_sec = max(0.0, seg.start - pad)
            end_sec = min(total_duration, seg.end + pad)
            start_sample = int(start_sec * 16000)
            end_sample = min(len(x_16k), int(end_sec * 16000))
            seg_audio = x_16k[start_sample:end_sample]

            if len(seg_audio) / 16000.0 >= 0.3:
                # Build CTC target from Turbo's transcription
                target_ids = []
                chars = []
                for ch in seg_text:
                    if ch in ctc_sym2id and not ch.isspace():
                        target_ids.append(ctc_sym2id[ch])
                        chars.append(ch)

                if target_ids and len(target_ids) > 0:
                    # Run Typhoon encoder + CTC head
                    fb = typhoon_fe(seg_audio.numpy(), sampling_rate=16000,
                                   return_attention_mask=True, return_tensors="pt")
                    mel_len = int(fb.attention_mask.sum())

                    with torch.no_grad():
                        h = typhoon_enc(
                            input_features=fb.input_features.to(device, typhoon_enc.dtype)
                        ).last_hidden_state[:, :(mel_len - 1) // 2 + 1]
                        logits = typhoon_head(h.float())
                        logp = torch.log_softmax(logits, dim=-1).cpu()

                    # Forced alignment
                    labels, scores = AF.forced_align(
                        logp, torch.tensor([target_ids], dtype=torch.int32), blank=0
                    )

                    # Extract character-level spans with 20ms precision
                    spans = []
                    lab = labels[0].tolist()
                    t, k = 0, 0
                    while t < len(lab) and k < len(chars):
                        if lab[t] == 0:
                            t += 1
                            continue
                        t0 = t
                        while t + 1 < len(lab) and lab[t + 1] == lab[t0]:
                            t += 1
                        s = round(max(0.0, t0 * 0.02) + start_sec, 3)
                        e = round(max(s + 0.02, (t + 1) * 0.02 + start_sec), 3)
                        spans.append({
                            "char": chars[k],
                            "start": s,
                            "end": e,
                            "conf": round(float(scores[0][t0:t + 1].mean()), 3)
                        })
                        k += 1
                        t += 1

                    # Only use CTC alignment if we got enough characters aligned (>= 80%)
                    if len(spans) >= len(chars) * 0.8:
                        # Group into natural words with PyThaiNLP
                        aligned_text = "".join(c["char"] for c in spans)
                        raw_words = thai_tokenize(aligned_text.replace(" ", ""))
                        idx = 0
                        for tw in raw_words:
                            grp = spans[idx:idx + len(tw)]
                            if not grp:
                                # Don't break! Continue with next word using estimated timing
                                continue
                            w_start = grp[0]["start"]
                            w_end = grp[-1]["end"]
                            # Cap trailing vowel duration
                            max_w_dur = max(0.50, len(tw) * 0.20)
                            if w_end - w_start > max_w_dur:
                                w_end = round(w_start + max_w_dur, 3)
                            all_words.append({
                                "word": tw,
                                "start": w_start,
                                "end": w_end,
                                "conf": round(sum(g["conf"] for g in grp) / len(grp), 3)
                            })
                            idx += len(tw)

                        ctc_success = True
                        ctc_aligned_count += 1

        except Exception as e:
            # CTC alignment failed for this segment, will fallback below
            pass

        # ===== FALLBACK: Use Turbo word timestamps if CTC failed =====
        if not ctc_success:
            turbo_fallback_count += 1
            # Use Turbo's own word-level timestamps with PyThaiNLP tokenization
            char_spans = []
            if seg.words:
                for w in seg.words:
                    clean = w.word.strip()
                    if not clean:
                        continue
                    dur = max(0.01, w.end - w.start)
                    ch_dur = dur / max(1, len(clean))
                    c_s = w.start
                    for ch in clean:
                        char_spans.append({
                            "char": ch,
                            "start": round(c_s, 3),
                            "end": round(c_s + ch_dur, 3),
                            "conf": round(w.probability, 3)
                        })
                        c_s += ch_dur
            else:
                clean = seg_text
                dur = max(0.01, seg.end - seg.start)
                ch_dur = dur / max(1, len(clean))
                c_s = seg.start
                for ch in clean:
                    char_spans.append({
                        "char": ch,
                        "start": round(c_s, 3),
                        "end": round(c_s + ch_dur, 3),
                        "conf": 0.90
                    })
                    c_s += ch_dur

            if char_spans:
                full_seg_text = "".join(c["char"] for c in char_spans)
                th_words = thai_tokenize(full_seg_text)
                idx = 0
                for tw in th_words:
                    grp = char_spans[idx:idx + len(tw)]
                    if not grp:
                        continue
                    w_start = grp[0]["start"]
                    w_end = grp[-1]["end"]
                    max_w_dur = max(0.50, len(tw) * 0.20)
                    if w_end - w_start > max_w_dur:
                        w_end = round(w_start + max_w_dur, 3)
                    all_words.append({
                        "word": tw,
                        "start": w_start,
                        "end": w_end,
                        "conf": round(sum(g["conf"] for g in grp) / len(grp), 3)
                    })
                    idx += len(tw)

    log_transcribe(f"[HYBRID] Step 2 done: CTC aligned {ctc_aligned_count} segments, Turbo fallback {turbo_fallback_count} segments")

    segments = form_dialogue_segments(all_words, max_chars_per_cue, max_pause_sec, min_cue_dur, time_offset=time_offset)
    full_text = " ".join(s["text"] for s in segments)
    return segments, full_text, total_duration

# ==========================================
# ENDPOINTS
# ==========================================
@app.on_event("startup")
def on_startup():
    logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
    log_transcribe("[READY] GPU Server พร้อมทำงาน (RTX 3090 Dual Engine: Typhoon CTC + Whisper Turbo) - สแตนด์บายรอคิวถอดเสียง...")

@app.get("/")
@app.get("/health")
async def health_check():
    with state_lock:
        now = time.time()
        idle_sec = max(0, int(now - last_active_time))
        rem_sec = max(0, IDLE_TIMEOUT_SECONDS - idle_sec) if active_jobs == 0 else IDLE_TIMEOUT_SECONDS
        busy = active_jobs > 0
        jobs_done = total_jobs_completed
        shutting = is_shutting_down

    return {
        "status": "ok",
        "service": "dual-engine-subtitle-api",
        "engines": {
            "thai": "typhoon-whisper-large-v3-ctc",
            "multilingual": "whisper-large-v3-turbo"
        },
        "gpu": GPU_NAME,
        "vram_gb": GPU_VRAM_GB,
        "idle_seconds": idle_sec,
        "auto_sleep_in_seconds": rem_sec,
        "auto_sleep_timeout": IDLE_TIMEOUT_SECONDS,
        "is_busy": busy,
        "active_jobs": active_jobs,
        "jobs_completed": jobs_done,
        "is_shutting_down": shutting
    }

@app.get("/touch")
@app.post("/touch")
async def touch_keep_alive():
    update_activity()
    return {
        "status": "ok",
        "message": "Keep-alive touch received. 4-hour idle timer reset.",
        "auto_sleep_in_seconds": IDLE_TIMEOUT_SECONDS
    }

@app.get("/logs")
def get_logs(lines: int = 60, mode: str = "transcribe"):
    target_path = TRANSCRIBE_LOG_PATH if mode == "transcribe" else SERVER_LOG_PATH
    if not os.path.exists(target_path):
        return {
            "logs": [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [READY] GPU Server พร้อมทำงาน (RTX 3090 Dual Engine) - สแตนด์บายรอคิวถอดเสียง..."
            ],
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    try:
        cmd = ["tail", "-n", str(min(500, max(10, lines))), target_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        raw_lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
        if mode == "transcribe" and not raw_lines:
            raw_lines = [
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [READY] GPU Server พร้อมทำงาน (RTX 3090 Dual Engine) - สแตนด์บายรอคิวถอดเสียง..."
            ]
        return {
            "logs": raw_lines,
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"]}

@app.get("/gpu")
def get_gpu_stats():
    try:
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "gpu_util_percent": int(float(parts[0])),
                "mem_used_mb": int(float(parts[1])),
                "mem_total_mb": int(float(parts[2])),
                "mem_used_gb": round(float(parts[1]) / 1024, 2),
                "mem_total_gb": round(float(parts[2]) / 1024, 2),
                "temp_c": int(float(parts[3])),
                "power_watts": round(float(parts[4]), 1) if len(parts) > 4 else 0
            }
    except Exception:
        pass
    return {
        "gpu_util_percent": 0,
        "mem_used_mb": 0,
        "mem_total_mb": 24576,
        "temp_c": 0,
        "power_watts": 0
    }

@app.post("/v1/system/stop")
async def manual_stop():
    threading.Thread(target=trigger_auto_sleep, daemon=True).start()
    return {
        "status": "stopping",
        "message": f"Stop command dispatched for instance {INSTANCE_ID}"
    }

def send_webhook_callback(url: str, payload: dict, retries: int = 3, delay: float = 3.0):
    try:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "AutoSub-GPU-Webhook/1.0"
            }
        )
        for attempt in range(1, retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    if resp.status in [200, 201, 202, 204]:
                        log_transcribe(f"[WEBHOOK SUCCESS] Callback sent to {url} (Attempt {attempt}, HTTP {resp.status})")
                        return True
            except Exception as ex:
                log_transcribe(f"[WEBHOOK RETRY {attempt}/{retries}] Failed to send to {url}: {ex}")
                if attempt < retries:
                    time.sleep(delay)
    except Exception as e:
        log_transcribe(f"[WEBHOOK FATAL] {e}")
    return False

def do_transcription_pipeline(
    tmp_path: str,
    orig_filename: str,
    model_name: Optional[str],
    language: Optional[str],
    response_format: Optional[str],
    temperature: Optional[float],
    prompt: Optional[str],
    isolate_vocals: Optional[bool],
    time_offset: Optional[float],
    start_ts: float,
    file_size_mb: float
):
    lang_code = (language or "th").strip().lower()
    req_model = (model_name or "auto").strip().lower()
    offset_val = -0.55 if time_offset is None else float(time_offset)

    if req_model in ["typhoon", "typhoon-ctc"]:
        engine_label = "Typhoon-Whisper-CTC" + (" + HDemucs" if isolate_vocals else "")
        log_transcribe(f"[JOB START] ไฟล์: {orig_filename} ({file_size_mb} MB) | ภาษา: {lang_code} | เอนจิน: {engine_label}")
        segments, combined_text, audio_dur = transcribe_with_typhoon(tmp_path, isolate_vocals=isolate_vocals)
        detected_lang = "th"
    elif req_model == "hybrid" or (req_model == "auto" and lang_code in ["th", "thai", "t1"]):
        engine_label = f"Hybrid (Turbo + Typhoon CTC Align, Offset: {offset_val}s)"
        log_transcribe(f"[JOB START] ไฟล์: {orig_filename} ({file_size_mb} MB) | ภาษา: {lang_code} | เอนจิน: {engine_label}")
        initial_prompt = prompt.strip() if prompt and prompt.strip() else None
        segments, combined_text, audio_dur = transcribe_hybrid_thai(
            tmp_path,
            initial_prompt=initial_prompt,
            temperature=temperature,
            time_offset=offset_val
        )
        detected_lang = "th"
    elif lang_code in ["th", "thai", "t1"]:
        engine_label = f"Faster-Whisper (large-v3-turbo) + Thai Dialogue Tuning (Offset: {offset_val}s)"
        log_transcribe(f"[JOB START] ไฟล์: {orig_filename} ({file_size_mb} MB) | ภาษา: {lang_code} | เอนจิน: {engine_label}")
        initial_prompt = prompt.strip() if prompt and prompt.strip() else None
        segments, combined_text, audio_dur = transcribe_with_turbo_thai(
            tmp_path,
            initial_prompt=initial_prompt,
            temperature=temperature,
            time_offset=offset_val
        )
        detected_lang = "th"
    else:
        engine_label = f"Faster-Whisper ({req_model or 'large-v3-turbo'})"
        log_transcribe(f"[JOB START] ไฟล์: {orig_filename} ({file_size_mb} MB) | ภาษา: {lang_code} | เอนจิน: {engine_label}")
        initial_prompt = prompt.strip() if prompt and prompt.strip() else None
        segments_gen, info = turbo_model.transcribe(
            tmp_path,
            language=lang_code if lang_code != "auto" else None,
            temperature=temperature,
            initial_prompt=initial_prompt,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        segments = []
        full_text_list = []
        for s in segments_gen:
            t = s.text.strip()
            if t:
                full_text_list.append(t)
                segments.append({
                    "id": s.id,
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": t
                })
        combined_text = " ".join(full_text_list)
        audio_dur = round(info.duration, 2)
        detected_lang = info.language

    infer_sec = round(time.time() - start_ts, 2)
    speedup = round(audio_dur / infer_sec, 1) if infer_sec > 0 else 0

    log_transcribe(f"[SUCCESS] ความยาวเสียง: {audio_dur}s | เวลาถอดเสียง: {infer_sec}s ({speedup}x เท่า) | จำนวนท่อน: {len(segments)} segments")
    if combined_text:
        preview = combined_text[:100] + ("..." if len(combined_text) > 100 else "")
        log_transcribe(f"[PREVIEW] \"{preview}\"")

    fmt = (response_format or "verbose_json").lower()
    if fmt == "vtt":
        vtt_lines = ["WEBVTT\n"]
        for s in segments:
            vtt_lines.append(f"{format_timestamp(s['start'], vtt=True)} --> {format_timestamp(s['end'], vtt=True)}")
            vtt_lines.append(f"{s['text']}\n")
        return PlainTextResponse("\n".join(vtt_lines), media_type="text/vtt; charset=utf-8")

    elif fmt == "srt":
        srt_lines = []
        for i, s in enumerate(segments, 1):
            srt_lines.append(str(i))
            srt_lines.append(f"{format_timestamp(s['start'], vtt=False)} --> {format_timestamp(s['end'], vtt=False)}")
            srt_lines.append(f"{s['text']}\n")
        return PlainTextResponse("\n".join(srt_lines), media_type="text/plain; charset=utf-8")

    elif fmt == "text":
        return PlainTextResponse(combined_text)

    # Default verbose_json
    return {
        "text": combined_text,
        "language": detected_lang,
        "duration": audio_dur,
        "segments": segments,
        "processing_time": infer_sec
    }

def async_webhook_worker(
    tmp_path: str,
    orig_filename: str,
    model_name: Optional[str],
    language: Optional[str],
    response_format: Optional[str],
    temperature: Optional[float],
    prompt: Optional[str],
    isolate_vocals: Optional[bool],
    time_offset: Optional[float],
    start_ts: float,
    file_size_mb: float,
    webhook_url: str,
    webhook_secret: Optional[str],
    custom_id: Optional[str]
):
    global active_jobs, total_jobs_completed, last_active_time
    token = current_file_ctx.set(orig_filename)
    try:
        res = do_transcription_pipeline(
            tmp_path, orig_filename, model_name, language, "verbose_json",
            temperature, prompt, isolate_vocals, time_offset, start_ts, file_size_mb
        )
        if isinstance(res, dict):
            # Format VTT
            vtt_lines = ["WEBVTT\n"]
            for s in res.get("segments", []):
                vtt_lines.append(f"{format_timestamp(s['start'], vtt=True)} --> {format_timestamp(s['end'], vtt=True)}")
                vtt_lines.append(f"{s['text']}\n")
            vtt_out = "\n".join(vtt_lines)

            # Format SRT
            srt_lines = []
            for i, s in enumerate(res.get("segments", []), 1):
                srt_lines.append(str(i))
                srt_lines.append(f"{format_timestamp(s['start'], vtt=False)} --> {format_timestamp(s['end'], vtt=False)}")
                srt_lines.append(f"{s['text']}\n")
            srt_out = "\n".join(srt_lines)

            payload = {
                "status": "completed",
                "custom_id": custom_id,
                "token": webhook_secret,
                "filename": orig_filename,
                "duration": res.get("duration", 0.0),
                "processing_time": res.get("processing_time", 0.0),
                "text": res.get("text", ""),
                "vtt": vtt_out,
                "srt": srt_out,
                "segments": res.get("segments", [])
            }
            send_webhook_callback(webhook_url, payload)
    except Exception as e:
        log_transcribe(f"[ASYNC WORKER ERROR] {e}")
        try:
            err_payload = {
                "status": "failed",
                "custom_id": custom_id,
                "token": webhook_secret,
                "error": str(e)
            }
            send_webhook_callback(webhook_url, err_payload, retries=1)
        except:
            pass
    finally:
        with state_lock:
            active_jobs = max(0, active_jobs - 1)
            total_jobs_completed += 1
            last_active_time = time.time()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        try:
            current_file_ctx.reset(token)
        except:
            pass

@app.post("/v1/audio/transcriptions")
def transcribe(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form("auto", alias="model"),
    language: Optional[str] = Form("th"),
    response_format: Optional[str] = Form("verbose_json"),
    temperature: Optional[float] = Form(0.0),
    prompt: Optional[str] = Form(None),
    isolate_vocals: Optional[bool] = Form(False),
    time_offset: Optional[float] = Form(-0.55),
    webhook_url: Optional[str] = Form(None),
    webhook_secret: Optional[str] = Form(None),
    custom_id: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    global active_jobs, total_jobs_completed, last_active_time
    start_ts = time.time()
    orig_filename = os.path.basename(file.filename or "audio.mp3")

    # Read uploaded file content to temporary file
    suffix = os.path.splitext(orig_filename)[1] or ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = file.file.read()
        tmp.write(content)
        tmp_path = tmp.name

    file_size_mb = round(len(content) / (1024 * 1024), 2)
    offset_val = -0.55 if time_offset is None else float(time_offset)

    # If webhook_url is supplied, execute asynchronously
    if webhook_url:
        with state_lock:
            active_jobs += 1
            last_active_time = start_ts
        log_transcribe(f"[WEBHOOK SUBMITTED] ไฟล์: {orig_filename} ({file_size_mb} MB) -> Webhook: {webhook_url} (custom_id: {custom_id})")
        background_tasks.add_task(
            async_webhook_worker,
            tmp_path, orig_filename, model_name, language, response_format,
            temperature, prompt, isolate_vocals, offset_val, start_ts, file_size_mb,
            webhook_url, webhook_secret, custom_id
        )
        return {
            "status": "queued",
            "message": "Task queued for asynchronous transcription. Result will be posted to webhook_url.",
            "custom_id": custom_id,
            "filename": orig_filename,
            "size_mb": file_size_mb
        }

    # Synchronous processing
    with state_lock:
        active_jobs += 1
        last_active_time = start_ts
    token = current_file_ctx.set(orig_filename)
    try:
        return do_transcription_pipeline(
            tmp_path, orig_filename, model_name, language, response_format,
            temperature, prompt, isolate_vocals, offset_val, start_ts, file_size_mb
        )
    except Exception as e:
        infer_sec = round(time.time() - start_ts, 2)
        log_transcribe(f"[ERROR] ถอดเสียงล้มเหลว ({infer_sec}s): {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        with state_lock:
            active_jobs = max(0, active_jobs - 1)
            total_jobs_completed += 1
            last_active_time = time.time()
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass
        try:
            current_file_ctx.reset(token)
        except:
            pass

