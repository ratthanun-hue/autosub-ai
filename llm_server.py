#!/usr/bin/env python3
"""
AutoSub-AI Central Master LLM Server (Step 6 Proofreader & Subtitle Translator)
Runs on Dedicated Master LLM Node (RTX A4000 16GB) on container port 10200.
Provides:
  1. High-speed contextual Thai subtitle proofreading via local Ollama (Qwen2.5-14B).
  2. Multi-language Subtitle Translation (Chinese, English, Korean, Japanese ⇄ Thai).
Delegated to by Dynamic Worker Nodes and Vidio backend.
"""

import os
import re
import time
import json
import math
import logging
import threading
import subprocess
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# Logging setup
os.makedirs("/root/whisper-server", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Master-LLM] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/root/whisper-server/llm_server.log", mode="a", encoding="utf-8")
    ]
)
logger = logging.getLogger("master_llm")

app = FastAPI(
    title="AutoSub-AI Central Master LLM Service",
    description="Contextual Thai Subtitle Refinement & Translation using Qwen2.5-14B",
    version="1.2.0"
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
DEFAULT_ENGINE = os.environ.get("LLM_ENGINE", "gemini").strip().lower()
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
DEFAULT_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

def resolve_gemini_api_key(req_key: Optional[str] = None) -> str:
    if req_key and req_key.strip():
        return req_key.strip()
    if DEFAULT_GEMINI_API_KEY:
        return DEFAULT_GEMINI_API_KEY
    for p in ["/root/whisper-server/.env", "/root/.env", os.path.expanduser("~/.env"), ".env"]:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            return line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
    try:
        req = urllib.request.Request("http://ph.cdnwatch.com/subtitle.php?action=get_cluster_config", headers={"User-Agent": "Master-LLM"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            cfg = json.loads(resp.read().decode("utf-8"))
            if cfg.get("gemini_api_key"):
                return cfg["gemini_api_key"].strip()
    except Exception:
        pass
    return ""

# Concurrency control: Allow concurrent batch refinements (Configurable via MAX_PARALLEL_BATCHES, default: 4)
MAX_PARALLEL_BATCHES = int(os.environ.get("MAX_PARALLEL_BATCHES", "4"))
LLM_SEMAPHORE = threading.Semaphore(MAX_PARALLEL_BATCHES)
active_requests = 0
active_requests_lock = threading.Lock()
total_completed_jobs = 0
total_completed_lock = threading.Lock()

class CueItem(BaseModel):
    id: int
    start: float
    end: float
    text: str

class RefineRequest(BaseModel):
    segments: List[Dict[str, Any]]
    drama_title: Optional[str] = None
    known_chars: Optional[List[str]] = None
    model: Optional[str] = DEFAULT_MODEL
    batch_size: Optional[int] = 25
    engine: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None

class AsyncRefineRequest(BaseModel):
    segments: List[Dict[str, Any]]
    drama_title: Optional[str] = None
    known_chars: Optional[List[str]] = None
    model: Optional[str] = DEFAULT_MODEL
    batch_size: Optional[int] = 25
    engine: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    webhook_url: str
    webhook_secret: Optional[str] = None
    token: Optional[str] = None
    custom_id: Optional[str] = None
    filename: Optional[str] = None
    duration: Optional[float] = 0.0
    processing_time_worker: Optional[float] = 0.0

class TranslateRequest(BaseModel):
    segments: Optional[List[Dict[str, Any]]] = None
    text: Optional[str] = None
    source_lang: Optional[str] = "auto"
    target_lang: Optional[str] = "th"
    drama_title: Optional[str] = None
    tone: Optional[str] = "natural" # e.g. "ancient_chinese_drama", "modern", "casual"
    model: Optional[str] = DEFAULT_MODEL
    batch_size: Optional[int] = 20

def check_ollama_status() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", headers={"User-Agent": "Master-LLM-Probe"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False

def warmup_model():
    """Keep model preloaded in GPU VRAM (keep_alive=-1) at all times."""
    try:
        payload = {
            "model": DEFAULT_MODEL,
            "keep_alive": -1,
            "messages": [{"role": "user", "content": "ping"}],
            "options": {"num_predict": 1}
        }
        req_body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=req_body,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            pass
        logger.info(f"Model {DEFAULT_MODEL} successfully warmed up in VRAM with keep_alive=-1")
    except Exception as e:
        logger.warning(f"Model warmup notice: {e}")

@app.on_event("startup")
def startup_event():
    t = threading.Thread(target=warmup_model, daemon=True)
    t.start()

@app.get("/", response_class=HTMLResponse)
def index_dashboard():
    html_content = """<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoSub-AI Central Master LLM Node</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #0a0e17;
            --card-bg: rgba(18, 26, 43, 0.75);
            --border-glow: rgba(56, 189, 248, 0.2);
            --primary: #38bdf8;
            --primary-glow: rgba(56, 189, 248, 0.4);
            --accent: #818cf8;
            --success: #34d399;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', 'Noto Sans Thai', sans-serif;
            background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0a0e17 70%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 30px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .container { width: 100%; max-width: 900px; }
        header { text-align: center; margin-bottom: 30px; }
        .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
            background: rgba(52, 211, 153, 0.15);
            border: 1px solid rgba(52, 211, 153, 0.4);
            color: var(--success);
            margin-bottom: 15px;
        }
        .dot { width: 8px; height: 8px; background: var(--success); border-radius: 50%; box-shadow: 0 0 10px var(--success); }
        h1 {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(135deg, #fff 0%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        p.subtitle { color: var(--text-muted); font-size: 0.95rem; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            border-radius: 16px;
            padding: 20px;
            transition: all 0.3s ease;
        }
        .card:hover {
            border-color: var(--primary);
            box-shadow: 0 8px 24px var(--primary-glow);
            transform: translateY(-2px);
        }
        .card-label { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
        .card-val { font-size: 1.3rem; font-weight: 700; color: #fff; }
        .card-val.green { color: var(--success); }
        .card-val.blue { color: var(--primary); }
        .main-panel {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-glow);
            border-radius: 20px;
            padding: 25px;
            margin-bottom: 25px;
        }
        h2 { font-size: 1.25rem; font-weight: 700; margin-bottom: 15px; color: var(--primary); }
        .form-group { margin-bottom: 15px; }
        label { display: block; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 600; }
        input, textarea {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            padding: 10px 14px;
            color: #fff;
            font-size: 0.95rem;
            font-family: inherit;
            outline: none;
            transition: border 0.2s;
        }
        input:focus, textarea:focus { border-color: var(--primary); }
        textarea { resize: vertical; min-height: 80px; }
        .btn {
            background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
            color: #fff;
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-size: 0.95rem;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        .btn:hover {
            box-shadow: 0 0 20px var(--primary-glow);
            transform: scale(1.02);
        }
        .btn-outline {
            background: transparent;
            border: 1px solid rgba(148, 163, 184, 0.3);
            color: var(--text-main);
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            transition: 0.2s;
        }
        .btn-outline:hover {
            border-color: var(--primary);
            color: var(--primary);
        }
        .links-bar {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            justify-content: center;
            margin-top: 20px;
        }
        #resultBox {
            margin-top: 15px;
            background: rgba(10, 14, 23, 0.8);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 10px;
            padding: 15px;
            font-family: monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            display: none;
            color: #38bdf8;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge"><div class="dot"></div> MASTER LLM ACTIVE (JAPAN DC 🇯🇵)</div>
            <h1>Central Master LLM Node</h1>
            <p class="subtitle">AutoSub-AI Step 6 Contextual Proofreading & Translation Cluster Service</p>
        </header>

        <div class="grid">
            <div class="card">
                <div class="card-label">AI Model</div>
                <div class="card-val blue" id="valModel">Qwen2.5-14B</div>
            </div>
            <div class="card">
                <div class="card-label">GPU Hardware</div>
                <div class="card-val" id="valGPU">RTX A4000 16GB</div>
            </div>
            <div class="card">
                <div class="card-label">Ollama Engine</div>
                <div class="card-val green" id="valOllama">CONNECTED</div>
            </div>
            <div class="card">
                <div class="card-label">Active / Completed</div>
                <div class="card-val" id="valJobs">0 / 0</div>
            </div>
        </div>

        <div class="main-panel">
            <h2>🧪 ทดสอบ Contextual Proofreading (Step 6)</h2>
            <div class="form-group">
                <label>ชื่อเรื่อง / ซีรีส์ (Drama Title)</label>
                <input type="text" id="dramaTitle" value="พรหมลิขิต" placeholder="เช่น พรหมลิขิต, บุพเพสันนิวาส">
            </div>
            <div class="form-group">
                <label>ข้อความซับไตเติลภาษาไทยที่ต้องการตรวจแก้ (Subtitle Text)</label>
                <textarea id="subText">สวัสดีเจ้าค่ะ แม่นาย</textarea>
            </div>
            <button class="btn" onclick="runProofread()">⚡ สั่ง Refine ผ่าน Qwen2.5-14B</button>
            <div id="resultBox"></div>
        </div>

        <div class="links-bar">
            <a class="btn-outline" href="/docs" target="_blank">📖 Swagger API Docs (/docs)</a>
            <a class="btn-outline" href="/health" target="_blank">🩺 Health Check JSON</a>
            <a class="btn-outline" href="/gpu" target="_blank">⚡ GPU Status JSON</a>
            <a class="btn-outline" href="/logs" target="_blank">📋 Server Logs</a>
        </div>
    </div>

    <script>
        async function fetchStats() {
            try {
                const res = await fetch('/health');
                if (res.ok) {
                    const data = await res.json();
                    document.getElementById('valModel').innerText = data.model || 'Qwen2.5-14B';
                    document.getElementById('valGPU').innerText = data.gpu || 'RTX A4000 16GB';
                    document.getElementById('valOllama').innerText = data.ollama_connected ? 'CONNECTED' : 'DISCONNECTED';
                    document.getElementById('valJobs').innerText = `${data.active_jobs || 0} / ${data.jobs_completed || 0}`;
                }
            } catch(e) {}
        }
        fetchStats();
        setInterval(fetchStats, 5000);

        async function runProofread() {
            const title = document.getElementById('dramaTitle').value.trim();
            const text = document.getElementById('subText').value.trim();
            const box = document.getElementById('resultBox');
            box.style.display = 'block';
            box.innerText = '⏳ กำลังประมวลผลผ่าน Qwen2.5-14B...';
            try {
                const t0 = performance.now();
                const res = await fetch('/v1/llm/refine', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        drama_title: title,
                        segments: [{ id: 1, start: 0.0, end: 2.0, text: text }]
                    })
                });
                const t1 = performance.now();
                const data = await res.json();
                box.innerText = `⏱️ ใช้เวลา: ${((t1 - t0) / 1000).toFixed(2)} วินาที\\n\\n` + JSON.stringify(data, null, 2);
            } catch (err) {
                box.innerText = '❌ เกิดข้อผิดพลาด: ' + err;
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.get("/health")
def health_check():
    ollama_ok = check_ollama_status()
    with active_requests_lock:
        req_count = active_requests
    with total_completed_lock:
        done_count = total_completed_jobs
    return {
        "status": "ok" if ollama_ok else "degraded",
        "service": "autosub-central-master-llm",
        "ollama_connected": ollama_ok,
        "ollama_url": OLLAMA_URL,
        "model": DEFAULT_MODEL,
        "gpu": "RTX A4000 16GB",
        "role": "master_llm",
        "vram_gb": 16,
        "active_jobs": req_count,
        "jobs_completed": done_count,
        "server_time": time.time()
    }

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
        "mem_total_mb": 16376,
        "mem_used_gb": 0,
        "mem_total_gb": 16.0,
        "temp_c": 0,
        "power_watts": 0
    }

@app.get("/logs")
def get_logs(lines: int = 50):
    log_path = "/root/whisper-server/llm_server.log"
    if not os.path.exists(log_path):
        return {
            "logs": [f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [READY] Master LLM Qwen2.5-14B server standby"],
            "server_time": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    try:
        cmd = ["tail", "-n", str(min(500, max(10, lines))), log_path]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        raw_lines = [l.strip() for l in res.stdout.splitlines() if l.strip()]
        return {"logs": raw_lines, "server_time": time.strftime("%Y-%m-%d %H:%M:%S")}
    except Exception as e:
        return {"logs": [f"Error reading logs: {e}"], "server_time": time.strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/v1/system/stop")
def system_stop():
    logger.info("Received /v1/system/stop request.")
    return {"status": "success", "message": "Master LLM stop signal acknowledged"}

def refine_subtitles_gemini(
    segments: List[Dict[str, Any]],
    drama_title: str = "ทั่วไป",
    known_chars: Optional[List[str]] = None,
    api_key: str = "",
    model: str = "gemini-3.5-flash-lite",
    batch_size: int = 35
) -> Tuple[List[Dict[str, Any]], int, float, str]:
    """
    Contextual Thai subtitle refinement powered by Google Gemini Flash API.
    Sends cues in batches (up to 35 cues each) with structured JSON diff return.
    Returns (refined_segments, total_corrected, duration_sec, status_info).
    Raises Exception if API call fails so caller can trigger seamless Ollama fallback.
    """
    t_start = time.time()
    char_str = ", ".join(known_chars[:15]) if known_chars else "ไม่ระบุ"
    system_prompt = (
        f"คุณคือ AI ผู้เชี่ยวชาญด้านการตรวจทานซับไตเติลภาษาไทย (Thai Subtitle Contextual Proofreader)\n"
        f"ภารกิจ: เกลาบริบทบทสนทนาและแก้ไขคำที่ระบบฟังเสียงพูด (ASR/Whisper) ฟังเพี้ยนหรือพ้องเสียง (Contextual Homophones)\n"
        f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
        f"รายชื่อตัวละครหลัก: {char_str}\n\n"
        "กฎเหล็กในการตรวจแก้:\n"
        "1. แก้ไขชื่อตัวละครที่ฟังเพี้ยนให้ตรงกับรายชื่อตัวละครหลักอย่างแม่นยำ (เช่น พี่ดนทร์/พี่เจด/พี่เจศ ให้แก้เป็น พี่ดล หรือ คุณภูดล, นี้ซิริ/เนซีรี ให้แก้เป็น เนตรศิริ ตามบริบทละคร)\n"
        "2. แก้ไขคำพ้องเสียงหรือคำที่ Whisper ฟังเพี้ยนจากเสียงพูด เช่น 'เต้นความ' -> 'แจ้งความ', 'จุดรวช' -> 'ตำรวจ', 'พลิกแฟร์ม' -> 'พลิกแฟ้ม', 'โมโมง' -> 'หมองมัว', 'สาปศูนย์' -> 'สาบสูญ'\n"
        "3. ข้อห้ามเด็ดขาด: ห้ามแต่งประโยคใหม่, ห้ามเติมคำลงท้าย (เช่น ห้ามเติม ค่ะ/ครับ/ฮะ ถ้าต้นฉบับไม่มี), และห้ามตัดทอนคำออก\n"
        "4. ข้อห้ามเรื่องรูปแบบ: ส่งเฉพาะประโยคหรือข้อความที่แก้ไขสมบูรณ์แล้วเท่านั้น ห้ามใส่เครื่องหมายลูกศร (-> หรือ →) หรือข้อความเปรียบเทียบเดิมเด็ดขาด\n"
        "5. กฎสำคัญด้านรูปแบบ (Diff-Only JSON): ส่งผลลัพธ์เป็น JSON Object เฉพาะ ID ที่มีการแก้ไขคำผิดเท่านั้น เช่น {\"2\": \"ข้อความที่แก้แล้ว\"} ห้ามใส่ ID ที่ถูกต้องอยู่แล้วลงมาในผลลัพธ์เด็ดขาด หากไม่มีคำผิดเลยให้ส่ง {}"
    )

    refined_segments = [dict(s) for s in segments]
    total_corrected = 0

    for i in range(0, len(refined_segments), batch_size):
        chunk = refined_segments[i:i + batch_size]
        cues_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}

        payload = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"จงตรวจแก้ซับไตเติลต่อไปนี้ และส่งคืนเฉพาะ ID ที่มีคำผิดในรูปแบบ JSON (หากไม่มีคำผิดให้ส่ง {{}}):\n{json.dumps(cues_dict, ensure_ascii=False, indent=2)}"}]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json"
            }
        }

        # Model trial order: requested model first, then gemini-3.5-flash-lite, then gemini-3.5-flash
        models_to_try = [model]
        for fallback_m in ["gemini-3.5-flash-lite", "gemini-3.5-flash"]:
            if fallback_m not in models_to_try:
                models_to_try.append(fallback_m)

        resp_data = None
        last_api_err = None
        for cur_model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{cur_model}:generateContent?key={api_key}"
            req_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_body,
                headers={"Content-Type": "application/json; charset=utf-8", "User-Agent": "AutoSub-GeminiEngine/1.0"}
            )
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(req, timeout=30.0) as resp:
                        resp_data = json.loads(resp.read().decode("utf-8"))
                        break
                except Exception as e:
                    last_api_err = e
                    if "429" in str(e) and attempt == 0:
                        time.sleep(1.5)
                        continue
                    logger.warning(f"[GEMINI] Call with model {cur_model} failed (attempt {attempt+1}): {e}.")
                    break
            if resp_data:
                break

        if not resp_data:
            raise last_api_err

        content = resp_data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "{}")
        corr_map = {}
        try:
            corr_map = json.loads(content)
        except Exception:
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                try:
                    corr_map = json.loads(m.group(0))
                except Exception:
                    pass

        if isinstance(corr_map, dict):
            for idx, s in enumerate(chunk, start=i):
                sid = str(s.get("id", idx))
                if sid in corr_map and isinstance(corr_map[sid], str) and corr_map[sid].strip():
                    new_text = corr_map[sid].strip()
                    if "->" in new_text or "→" in new_text:
                        parts = re.split(r'\s*(?:->|→)\s*', new_text)
                        new_text = parts[-1].strip()
                    new_text = re.sub(r'^[\'"]|[\'"]$', '', new_text).strip()
                    if new_text and new_text != s.get("text", ""):
                        s["text"] = new_text
                        total_corrected += 1

    dur = round(time.time() - t_start, 2)
    return refined_segments, total_corrected, dur, "gemini_success"

# ==============================================================================
# 1. CONTEXTUAL THAI PROOFREADING (Step 6)
# ==============================================================================
@app.post("/v1/llm/refine")
def refine_subtitles(req_data: RefineRequest):
    global active_requests, total_completed_jobs
    segments = req_data.segments
    if not segments:
        return {"status": "success", "corrected_count": 0, "segments": [], "duration_sec": 0.0}

    drama_title = req_data.drama_title or "ทั่วไป"
    known_chars = req_data.known_chars or []
    engine = (req_data.engine or DEFAULT_ENGINE or "gemini").strip().lower()
    gemini_key = resolve_gemini_api_key(req_data.gemini_api_key)
    gemini_mod = req_data.gemini_model or DEFAULT_GEMINI_MODEL

    # Route 1: Google Gemini Flash Engine (Pilot 1-Day Active)
    if engine == "gemini" and gemini_key:
        try:
            logger.info(f"[GEMINI STEP 6] Refining {len(segments)} cues via {gemini_mod} for drama '{drama_title}'...")
            ref_segs, corr_cnt, dur, _ = refine_subtitles_gemini(
                segments=segments,
                drama_title=drama_title,
                known_chars=known_chars,
                api_key=gemini_key,
                model=gemini_mod
            )
            with total_completed_lock:
                total_completed_jobs += 1
            logger.info(f"[GEMINI SUCCESS] Refined {corr_cnt} cues in {dur}s (Engine: {gemini_mod}).")
            return {
                "status": "success",
                "engine_used": gemini_mod,
                "corrected_count": corr_cnt,
                "segments": ref_segs,
                "duration_sec": dur
            }
        except Exception as g_err:
            logger.warning(f"[GEMINI NOTICE] Gemini call failed ({g_err}). Auto-Fallback to Local Ollama...")

    # Route 2: Local Ollama (Qwen2.5-14B) - Default / Fallback
    model = req_data.model or DEFAULT_MODEL
    # Auto-upgrade to 14B on Master Node when 7b is requested by older workers
    if model in ("qwen2.5:7b", "qwen2.5", ""):
        model = DEFAULT_MODEL
    batch_size = max(10, min(req_data.batch_size or 25, 40))

    if not check_ollama_status():
        logger.warning(f"Ollama is unreachable at {OLLAMA_URL}. Returning original cues as fallback.")
        return {
            "status": "fallback",
            "message": "Ollama service unavailable on Master Node",
            "corrected_count": 0,
            "segments": segments,
            "duration_sec": 0.0
        }

    acquired = LLM_SEMAPHORE.acquire(timeout=600.0)
    if not acquired:
        logger.warning(f"Master LLM semaphore timeout after 600s for drama '{drama_title}'. Returning original cues.")
        return {
            "status": "busy_fallback",
            "message": "Master LLM queue busy",
            "corrected_count": 0,
            "segments": segments,
            "duration_sec": 0.0
        }

    with active_requests_lock:
        active_requests += 1

    t_start = time.time()
    logger.info(f"Processing LLM refinement request: {len(segments)} cues, drama='{drama_title}', chars={len(known_chars)}")

    try:
        char_str = ", ".join(known_chars[:12]) if known_chars else "ไม่ระบุ"
        system_prompt = (
            f"คุณคือ AI ผู้เชี่ยวชาญด้านการตรวจทานซับไตเติลภาษาไทย (Thai Subtitle Contextual Proofreader)\n"
            f"ภารกิจ: เกลาบริบทบทสนทนาและแก้ไขคำที่ระบบฟังเสียงพูด (ASR/Whisper) ฟังเพี้ยนหรือพ้องเสียง (Contextual Homophones)\n"
            f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
            f"รายชื่อตัวละครหลัก: {char_str}\n\n"
            "กฎเหล็กในการตรวจแก้:\n"
            "1. แก้ไขชื่อตัวละครที่ฟังเพี้ยนให้ตรงกับรายชื่อตัวละครหลัก (เช่น หากได้ยิน พี่ดนทร์/พี่ผู้ชม ให้แก้เป็น พี่ดล หรือ คุณภูดล, นี้ซิริ/เนซีรี ให้แก้เป็น เนตรศิริ ตามบริบทละคร)\n"
            "2. แก้ไขคำพ้องเสียงหรือคำที่ Whisper ฟังเพี้ยนจากเสียงพูด เช่น 'เต้นความ' แก้เป็น 'แจ้งความ', 'จุดรวช' แก้เป็น 'ตำรวจ', 'พลิกแฟร์ม' แก้เป็น 'พลิกแฟ้ม', 'โมโมง' แก้เป็น 'หมองมัว', 'สาปศูนย์' แก้เป็น 'สาบสูญ'\n"
            "3. ข้อห้ามเด็ดขาด: ห้ามแต่งประโยคใหม่, ห้ามเติมคำลงท้าย (เช่น ห้ามเติม ค่ะ/ครับ/ฮะ ถ้าต้นฉบับไม่มี), และห้ามตัดทอนคำออก\n"
            "4. ข้อห้ามเรื่องรูปแบบ: ให้ส่งเฉพาะประโยคหรือข้อความที่แก้ไขสมบูรณ์แล้วเท่านั้น ห้ามใส่เครื่องหมายลูกศร (-> หรือ →) หรือข้อความเปรียบเทียบเดิมเด็ดขาด เช่น ให้ส่ง 'ตำรวจ' ห้ามส่ง 'จุดรวช -> ตำรวจ'\n"
            "5. กฎสำคัญด้านความเร็ว (Diff-Only): ส่งผลลัพธ์เป็น JSON Object เฉพาะ ID ที่มีการแก้ไขคำผิดเท่านั้น เช่น {\"2\": \"ข้อความที่แก้แล้ว\"} ห้ามใส่ ID ที่ถูกต้องอยู่แล้วลงมาในผลลัพธ์เด็ดขาด หากไม่มีคำผิดเลยให้ส่ง {}"
        )

        refined_segments = [dict(s) for s in segments]
        total_corrected = 0

        for i in range(0, len(refined_segments), batch_size):
            chunk = refined_segments[i:i + batch_size]
            cues_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}

            try:
                payload = {
                    "model": model,
                    "format": "json",
                    "stream": False,
                    "keep_alive": -1,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024
                    },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"จงตรวจแก้ซับไตเติลต่อไปนี้ และส่งคืนเฉพาะ ID ที่มีคำผิด (หากไม่มีคำผิดให้ส่ง {{}}):\n{json.dumps(cues_dict, ensure_ascii=False, indent=2)}"}
                    ]
                }

                req_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/chat",
                    data=req_body,
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )

                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    content = resp_json.get("message", {}).get("content", "").strip()

                    corr_map = {}
                    try:
                        corr_map = json.loads(content)
                    except Exception:
                        m = re.search(r'\{.*\}', content, re.DOTALL)
                        if m:
                            corr_map = json.loads(m.group(0))

                    if isinstance(corr_map, dict):
                        for idx, s in enumerate(chunk, start=i):
                            sid = str(s.get("id", idx))
                            if sid in corr_map and isinstance(corr_map[sid], str) and corr_map[sid].strip():
                                new_text = corr_map[sid].strip()
                                if "->" in new_text or "→" in new_text:
                                    parts = re.split(r'\s*(?:->|→)\s*', new_text)
                                    new_text = parts[-1].strip()
                                new_text = re.sub(r'^[\'"]|[\'"]$', '', new_text).strip()
                                if new_text != s.get("text", ""):
                                    s["text"] = new_text
                                    total_corrected += 1
            except Exception as batch_err:
                logger.warning(f"Batch {i // batch_size + 1} refinement warning: {batch_err}. Keeping original cues.")
                continue

        dur = round(time.time() - t_start, 2)
        with total_completed_lock:
            total_completed_jobs += 1
        logger.info(f"Refinement completed: {total_corrected} cues corrected in {dur}s for drama '{drama_title}'.")
        return {
            "status": "success",
            "corrected_count": total_corrected,
            "segments": refined_segments,
            "duration_sec": dur
        }

    except Exception as ex:
        dur = round(time.time() - t_start, 2)
        logger.error(f"Error during LLM refinement: {ex}")
        return {
            "status": "error_fallback",
            "message": str(ex),
            "corrected_count": 0,
            "segments": segments,
            "duration_sec": dur
        }
    finally:
        with active_requests_lock:
            active_requests -= 1
        LLM_SEMAPHORE.release()

# ==============================================================================
# 1.1 ASYNC CONTEXTUAL THAI PROOFREADING (Async Webhook Delegation)
# ==============================================================================
def format_timestamp(seconds: float, vtt: bool = False) -> str:
    hours = math.floor(seconds / 3600)
    minutes = math.floor((seconds % 3600) / 60)
    secs = math.floor(seconds % 60)
    msecs = math.floor((seconds - math.floor(seconds)) * 1000)
    sep = "." if vtt else ","
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{msecs:03d}"

def format_thai_subtitle_line_wrap(text: str, max_cpl: int = 38, max_lines: int = 2) -> str:
    """Soft-wraps Thai subtitle cues at safe word boundaries (CPL <= 38, max 2 lines)."""
    if not text or not str(text).strip():
        return text or ""
    clean_text = str(text).strip()
    if len(clean_text) <= max_cpl and "\n" not in clean_text:
        return clean_text
    existing_lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
    if 1 < len(existing_lines) <= max_lines and all(len(l) <= max_cpl for l in existing_lines):
        return "\n".join(existing_lines)
    flat_text = " ".join(clean_text.split())
    if len(flat_text) <= max_cpl:
        return flat_text
    try:
        from pythainlp.tokenize import word_tokenize
        tokens = word_tokenize(flat_text, engine="newmm")
    except Exception:
        tokens = [flat_text]
    if not tokens or len(tokens) <= 1:
        return flat_text
    best_idx = -1
    best_score = float("inf")
    total_len = sum(len(t) for t in tokens)
    cur_len = 0
    for i in range(len(tokens) - 1):
        cur_len += len(tokens[i])
        rem_len = total_len - cur_len
        penalty = 0
        if cur_len > max_cpl:
            penalty += (cur_len - max_cpl) * 100
        if rem_len > max_cpl:
            penalty += (rem_len - max_cpl) * 50
        score = abs(cur_len - rem_len) + penalty
        if score < best_score:
            best_score = score
            best_idx = i + 1
    if best_idx <= 0 or best_idx >= len(tokens):
        best_idx = len(tokens) // 2
    line1 = "".join(tokens[:best_idx]).strip()
    line2 = "".join(tokens[best_idx:]).strip()
    return f"{line1}\n{line2}" if line2 else line1

def send_webhook_callback(url: str, payload: dict, retries: int = 3, delay: float = 3.0) -> bool:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json; charset=utf-8', 'User-Agent': 'AutoSub-MasterLLM-Webhook/1.0'}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status in [200, 201, 202, 204]:
                    logger.info(f"[WEBHOOK SUCCESS] Callback sent to {url} (Attempt {attempt}, HTTP {resp.status})")
                    return True
        except Exception as ex:
            logger.warning(f"[WEBHOOK RETRY {attempt}/{retries}] Failed to send to {url}: {ex}")
            if attempt < retries:
                time.sleep(delay)
    return False

def _send_fallback_webhook(job: AsyncRefineRequest, reason: str):
    try:
        custom_id = job.custom_id or "unknown"
        logger.warning(f"Sending fallback webhook for custom_id='{custom_id}' due to: {reason}")
        # Format VTT with Step 5 cues
        vtt_lines = ["WEBVTT\n"]
        for s in job.segments:
            vtt_lines.append(f"{format_timestamp(s['start'], vtt=True)} --> {format_timestamp(s['end'], vtt=True)}")
            spk = s.get("speaker")
            txt = format_thai_subtitle_line_wrap(s.get("text", ""))
            if spk:
                vtt_lines.append(f"<v {spk}>{txt}\n")
            else:
                vtt_lines.append(f"{txt}\n")
        vtt_out = "\n".join(vtt_lines)
        combined_text = " ".join(s.get("text", "") for s in job.segments)

        webhook_payload = {
            "status": "completed",
            "custom_id": custom_id,
            "token": job.webhook_secret or job.token,
            "filename": job.filename or f"sub_{custom_id}.mp3",
            "duration": job.duration or 0.0,
            "processing_time": job.processing_time_worker or 0.0,
            "text": combined_text,
            "vtt": vtt_out,
            "segments": job.segments,
            "step6_refined": False,
            "notice": reason
        }
        send_webhook_callback(job.webhook_url, webhook_payload)
    except Exception as e:
        logger.error(f"Failed to send fallback webhook: {e}")

def _process_async_refine(job: AsyncRefineRequest):
    global active_requests, total_completed_jobs
    t_start = time.time()
    custom_id = job.custom_id or "unknown"
    logger.info(f"[ASYNC START] Processing Thai refinement for custom_id='{custom_id}', drama='{job.drama_title}', cues={len(job.segments)}")
    
    with active_requests_lock:
        active_requests += 1

    # Generous queue timeout (30 mins) so no job is dropped
    acquired = LLM_SEMAPHORE.acquire(timeout=1800.0)
    if not acquired:
        logger.error(f"[ASYNC TIMEOUT] Semaphore queue timeout for custom_id='{custom_id}'")
        with active_requests_lock:
            active_requests -= 1
        _send_fallback_webhook(job, "LLM queue busy")
        return

    try:
        segments = [dict(s) for s in job.segments]
        drama_title = job.drama_title or "ทั่วไป"
        known_chars = job.known_chars or []
        model = job.model or DEFAULT_MODEL
        if model in ("qwen2.5:7b", "qwen2.5", ""):
            model = DEFAULT_MODEL
        batch_size = max(10, min(job.batch_size or 25, 40))

        char_str = ", ".join(known_chars[:12]) if known_chars else "ไม่ระบุ"
        total_corrected = 0

        engine = (job.engine or DEFAULT_ENGINE or "gemini").strip().lower()
        gemini_key = resolve_gemini_api_key(job.gemini_api_key)
        gemini_mod = job.gemini_model or DEFAULT_GEMINI_MODEL
        gemini_success = False

        # Route 1: Google Gemini Flash Engine (Pilot 1-Day Active)
        if engine == "gemini" and gemini_key:
            try:
                logger.info(f"[GEMINI ASYNC] Refining {len(segments)} cues via {gemini_mod} for '{custom_id}'...")
                ref_segs, corr_cnt, g_dur, _ = refine_subtitles_gemini(
                    segments=segments,
                    drama_title=drama_title,
                    known_chars=known_chars,
                    api_key=gemini_key,
                    model=gemini_mod
                )
                segments = ref_segs
                total_corrected = corr_cnt
                gemini_success = True
                logger.info(f"[GEMINI ASYNC SUCCESS] Refined {corr_cnt} cues in {g_dur}s via {gemini_mod} for '{custom_id}'.")
            except Exception as g_err:
                logger.warning(f"[GEMINI ASYNC NOTICE] Gemini failed ({g_err}). Auto-Fallback to Local Ollama...")

        # Route 2: Local Ollama (Qwen2.5-14B) - Fallback or when engine=ollama
        if not gemini_success:
            system_prompt = (
                f"คุณคือ AI ผู้เชี่ยวชาญด้านการตรวจทานซับไตเติลภาษาไทย (Thai Subtitle Contextual Proofreader)\n"
                f"ภารกิจ: เกลาบริบทบทสนทนาและแก้ไขคำที่ระบบฟังเสียงพูด (ASR/Whisper) ฟังเพี้ยนหรือพ้องเสียง (Contextual Homophones)\n"
                f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
                f"รายชื่อตัวละครหลัก: {char_str}\n\n"
                "กฎเหล็กในการตรวจแก้:\n"
                "1. แก้ไขชื่อตัวละครที่ฟังเพี้ยนให้ตรงกับรายชื่อตัวละครหลัก (เช่น หากได้ยิน พี่ดนทร์/พี่ผู้ชม ให้แก้เป็น พี่ดล หรือ คุณภูดล, นี้ซิริ/เนซีรี ให้แก้เป็น เนตรศิริ ตามบริบทละคร)\n"
                "2. แก้ไขคำพ้องเสียงหรือคำที่ Whisper ฟังเพี้ยนจากเสียงพูด เช่น 'เต้นความ' แก้เป็น 'แจ้งความ', 'จุดรวช' แก้เป็น 'ตำรวจ', 'พลิกแฟร์ม' แก้เป็น 'พลิกแฟ้ม', 'โมโมง' แก้เป็น 'หมองมัว', 'สาปศูนย์' แก้เป็น 'สาบสูญ'\n"
                "3. ข้อห้ามเด็ดขาด: ห้ามแต่งประโยคใหม่, ห้ามเติมคำลงท้าย (เช่น ห้ามเติม ค่ะ/ครับ/ฮะ ถ้าต้นฉบับไม่มี), และห้ามตัดทอนคำออก\n"
                "4. ข้อห้ามเรื่องรูปแบบ: ให้ส่งเฉพาะประโยคหรือข้อความที่แก้ไขสมบูรณ์แล้วเท่านั้น ห้ามใส่เครื่องหมายลูกศร (-> หรือ →) หรือข้อความเปรียบเทียบเดิมเด็ดขาด เช่น ให้ส่ง 'ตำรวจ' ห้ามส่ง 'จุดรวช -> ตำรวจ'\n"
                "5. กฎสำคัญด้านความเร็ว (Diff-Only): ส่งผลลัพธ์เป็น JSON Object เฉพาะ ID ที่มีการแก้ไขคำผิดเท่านั้น เช่น {{\"2\": \"ข้อความที่แก้แล้ว\"}} ห้ามใส่ ID ที่ถูกต้องอยู่แล้วลงมาในผลลัพธ์เด็ดขาด หากไม่มีคำผิดเลยให้ส่ง {{}}"
            )

            total_corrected = 0
            for i in range(0, len(segments), batch_size):
                chunk = segments[i:i + batch_size]
                cues_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}

                try:
                    payload = {
                        "model": model,
                        "format": "json",
                        "stream": False,
                        "keep_alive": -1,
                        "options": {"temperature": 0.1, "num_predict": 1024},
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"จงตรวจแก้ซับไตเติลต่อไปนี้ และส่งคืนเฉพาะ ID ที่มีคำผิด (หากไม่มีคำผิดให้ส่ง {{}}):\n{json.dumps(cues_dict, ensure_ascii=False, indent=2)}"}
                        ]
                    }
                    req_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    req = urllib.request.Request(
                        f"{OLLAMA_URL}/api/chat",
                        data=req_body,
                        headers={"Content-Type": "application/json; charset=utf-8"}
                    )
                    with urllib.request.urlopen(req, timeout=120.0) as resp:
                        resp_json = json.loads(resp.read().decode("utf-8"))
                        content = resp_json.get("message", {}).get("content", "").strip()

                        corr_map = {}
                        try:
                            corr_map = json.loads(content)
                        except Exception:
                            m = re.search(r'\{.*\}', content, re.DOTALL)
                            if m:
                                corr_map = json.loads(m.group(0))

                        if isinstance(corr_map, dict):
                            for idx, s in enumerate(chunk, start=i):
                                sid = str(s.get("id", idx))
                                if sid in corr_map and isinstance(corr_map[sid], str) and corr_map[sid].strip():
                                    new_text = corr_map[sid].strip()
                                    if "->" in new_text or "→" in new_text:
                                        parts = re.split(r'\s*(?:->|→)\s*', new_text)
                                        new_text = parts[-1].strip()
                                    new_text = re.sub(r'^[\'"]|[\'"]$', '', new_text).strip()
                                    if new_text != s.get("text", ""):
                                        s["text"] = new_text
                                        total_corrected += 1
                except Exception as b_err:
                    logger.warning(f"Batch {i // batch_size + 1} async refine warning: {b_err}. Keeping original cues.")
                    continue

        dur = round(time.time() - t_start, 2)
        total_proc = round((job.processing_time_worker or 0.0) + dur, 2)
        logger.info(f"[ASYNC SUCCESS] Refined {total_corrected} cues in {dur}s for custom_id='{custom_id}' (Total time: {total_proc}s). Delivering webhook...")

        # Format VTT
        vtt_lines = ["WEBVTT\n"]
        for s in segments:
            vtt_lines.append(f"{format_timestamp(s['start'], vtt=True)} --> {format_timestamp(s['end'], vtt=True)}")
            spk = s.get("speaker")
            txt = format_thai_subtitle_line_wrap(s.get("text", ""))
            if spk:
                vtt_lines.append(f"<v {spk}>{txt}\n")
            else:
                vtt_lines.append(f"{txt}\n")
        vtt_out = "\n".join(vtt_lines)

        # Format SRT
        srt_lines = []
        for i_idx, s in enumerate(segments, 1):
            srt_lines.append(str(i_idx))
            srt_lines.append(f"{format_timestamp(s['start'], vtt=False)} --> {format_timestamp(s['end'], vtt=False)}")
            spk = s.get("speaker")
            txt = format_thai_subtitle_line_wrap(s.get("text", ""))
            if spk:
                srt_lines.append(f"[{spk}]: {txt}\n")
            else:
                srt_lines.append(f"{txt}\n")
        srt_out = "\n".join(srt_lines)
        combined_text = " ".join(s.get("text", "") for s in segments)

        webhook_payload = {
            "status": "completed",
            "custom_id": custom_id,
            "token": job.webhook_secret or job.token,
            "filename": job.filename or f"sub_{custom_id}.mp3",
            "duration": job.duration or 0.0,
            "processing_time": total_proc,
            "text": combined_text,
            "vtt": vtt_out,
            "srt": srt_out,
            "segments": segments,
            "step6_refined": True,
            "corrected_count": total_corrected
        }
        send_webhook_callback(job.webhook_url, webhook_payload)

        with total_completed_lock:
            total_completed_jobs += 1

    except Exception as ex:
        logger.error(f"[ASYNC ERROR] Unexpected error for custom_id='{custom_id}': {ex}")
        _send_fallback_webhook(job, str(ex))
    finally:
        with active_requests_lock:
            active_requests -= 1
        LLM_SEMAPHORE.release()

@app.post("/v1/llm/refine_async")
def refine_subtitles_async(req_data: AsyncRefineRequest):
    if not req_data.segments:
        raise HTTPException(status_code=400, detail="segments list cannot be empty")
    if not req_data.webhook_url:
        raise HTTPException(status_code=400, detail="webhook_url is required for async refinement")

    logger.info(f"Queuing async refinement for custom_id='{req_data.custom_id}', cues={len(req_data.segments)}")
    t = threading.Thread(target=_process_async_refine, args=(req_data,), daemon=True)
    t.start()

    return {
        "status": "queued",
        "custom_id": req_data.custom_id,
        "message": "Thai subtitle refinement queued asynchronously. Result will be posted to webhook.",
        "cues_count": len(req_data.segments),
        "drama_title": req_data.drama_title
    }

# ==============================================================================
# 2. MULTILINGUAL SUBTITLE TRANSLATION
# ==============================================================================
@app.post("/v1/llm/translate")
def translate_subtitles(req_data: TranslateRequest):
    global active_requests, total_completed_jobs
    model = req_data.model or DEFAULT_MODEL
    if model in ("qwen2.5:7b", "qwen2.5", ""):
        model = DEFAULT_MODEL
    source_lang = req_data.source_lang or "auto"
    target_lang = req_data.target_lang or "th"
    drama_title = req_data.drama_title or "ทั่วไป"
    tone = req_data.tone or "natural"
    batch_size = max(5, min(req_data.batch_size or 20, 30))

    if not check_ollama_status():
        return {"status": "fallback", "message": "Ollama service unavailable on Master Node"}

    acquired = LLM_SEMAPHORE.acquire(timeout=60.0)
    if not acquired:
        return {"status": "busy_fallback", "message": "Master LLM queue busy"}

    with active_requests_lock:
        active_requests += 1

    t_start = time.time()

    try:
        tone_instruction = ""
        if "ancient" in tone or "chinese" in tone or "ยุค" in tone or "โบราณ" in tone:
            tone_instruction = "ใช้ภาษาไทยโบราณที่สละสลวย เหมาะกับซีรีส์ย้อนยุค/กำลังภายใน ใช้คำสรรพนามและลำดับยศที่ถูกต้อง (เช่น ข้า, ท่าน, เจ้า, องค์ชาย, แม่ทัพ)"
        elif "formal" in tone:
            tone_instruction = "ใช้ภาษาทางการ สุภาพ ถูกหลักไวยากรณ์"
        else:
            tone_instruction = "ใช้ภาษาพูดที่เป็นธรรมชาติ เหมาะกับบทสนทนาภาพยนตร์และละคร"

        # Case A: Plain Text Translation
        if req_data.text and not req_data.segments:
            prompt = (
                f"You are a professional literary translator.\n"
                f"Translate the following text from {source_lang} to {target_lang}.\n"
                f"Context: {drama_title}\n"
                f"Tone style: {tone_instruction}\n\n"
                f"Output ONLY the translated text without extra explanation."
            )
            payload = {
                "model": model,
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.2, "num_predict": 2048},
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": req_data.text}
                ]
            }
            req_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=req_body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120.0) as resp:
                resp_json = json.loads(resp.read().decode("utf-8"))
                translated = resp_json.get("message", {}).get("content", "").strip()

            dur = round(time.time() - t_start, 2)
            return {
                "status": "success",
                "translated_text": translated,
                "duration_sec": dur
            }

        # Case B: Subtitle Segments Translation
        segments = req_data.segments or []
        if not segments:
            return {"status": "success", "segments": [], "duration_sec": 0.0}

        translated_segments = [dict(s) for s in segments]
        system_prompt = (
            f"You are an expert subtitle translator.\n"
            f"Translate dialogue subtitle cues from {source_lang} into natural, engaging {target_lang}.\n"
            f"Context: Drama '{drama_title}'\n"
            f"Tone style: {tone_instruction}\n"
            f"Important Rules:\n"
            f"1. Keep subtitles concise and readable on screen.\n"
            f"2. Retain consistent character pronouns and emotional nuance across the scene.\n"
            f"3. Return ONLY a valid JSON object mapping ID to translated text: {{\"ID\": \"translated text\"}}"
        )

        total_translated = 0
        for i in range(0, len(translated_segments), batch_size):
            chunk = translated_segments[i:i + batch_size]
            cues_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}

            try:
                payload = {
                    "model": model,
                    "format": "json",
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": 0.2, "num_predict": 2048},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Translate these subtitle cues:\n{json.dumps(cues_dict, ensure_ascii=False, indent=2)}"}
                    ]
                }
                req_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                req = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=req_body, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=120.0) as resp:
                    resp_json = json.loads(resp.read().decode("utf-8"))
                    content = resp_json.get("message", {}).get("content", "").strip()
                    trans_map = {}
                    try:
                        trans_map = json.loads(content)
                    except Exception:
                        m = re.search(r'\{.*\}', content, re.DOTALL)
                        if m:
                            trans_map = json.loads(m.group(0))

                    if isinstance(trans_map, dict):
                        for idx, s in enumerate(chunk, start=i):
                            sid = str(s.get("id", idx))
                            if sid in trans_map and isinstance(trans_map[sid], str) and trans_map[sid].strip():
                                s["text"] = trans_map[sid].strip()
                                total_translated += 1
            except Exception as b_err:
                logger.warning(f"Translation batch error: {b_err}")
                continue

        dur = round(time.time() - t_start, 2)
        return {
            "status": "success",
            "translated_count": total_translated,
            "segments": translated_segments,
            "duration_sec": dur
        }

    except Exception as ex:
        dur = round(time.time() - t_start, 2)
        logger.error(f"Error during translation: {ex}")
        return {"status": "error", "message": str(ex), "duration_sec": dur}
    finally:
        with active_requests_lock:
            active_requests -= 1
        LLM_SEMAPHORE.release()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("LLM_PORT", 10200))
    logger.info(f"Starting AutoSub-AI Master LLM Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
