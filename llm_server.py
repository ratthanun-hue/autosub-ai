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
STAGE1_MODEL = os.environ.get("STAGE1_MODEL", "scb10x/llama3.1-typhoon2-8b-instruct:latest")
STAGE2_MODEL = os.environ.get("STAGE2_MODEL", "qwen2.5:14b-instruct-q5_K_M")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", STAGE2_MODEL)
DEFAULT_ENGINE = os.environ.get("LLM_ENGINE", "ollama").strip().lower()
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

# Concurrency control: Single dedicated stream (MAX_PARALLEL_BATCHES=1) to prevent GPU contention on V100
MAX_PARALLEL_BATCHES = int(os.environ.get("MAX_PARALLEL_BATCHES", "1"))
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
    stage1_model: Optional[str] = None
    stage2_model: Optional[str] = None
    batch_size: Optional[int] = 25
    engine: Optional[str] = None
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None

class AsyncRefineRequest(BaseModel):
    segments: List[Dict[str, Any]]
    drama_title: Optional[str] = None
    known_chars: Optional[List[str]] = None
    model: Optional[str] = DEFAULT_MODEL
    stage1_model: Optional[str] = None
    stage2_model: Optional[str] = None
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
    """Keep both Stage 1 (Typhoon 2.1 Gemma3 12B) and Stage 2 (Qwen 14B Q5_K_M) preloaded in GPU VRAM (keep_alive=-1)."""
    for m in [STAGE1_MODEL, STAGE2_MODEL]:
        try:
            payload = {
                "model": m,
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
            logger.info(f"Model '{m}' successfully warmed up in VRAM with keep_alive=-1")
        except Exception as e:
            logger.warning(f"Model warmup notice for '{m}': {e}")

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
                <div class="card-val blue" id="valModel">Qwen2.5-32B</div>
            </div>
            <div class="card">
                <div class="card-label">GPU Hardware</div>
                <div class="card-val" id="valGPU">Tesla V100 32GB</div>
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
            <button class="btn" onclick="runProofread()">⚡ สั่ง Refine ผ่าน Qwen2.5-32B</button>
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

def get_gpu_info():
    try:
        cmd = ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            name = parts[0]
            total_mb = float(parts[1]) if len(parts) > 1 and parts[1].replace('.','',1).isdigit() else 32768
            return name, round(total_mb / 1024, 1)
    except Exception:
        pass
    return "Tesla V100 32GB", 32.0

@app.get("/health")
def health_check():
    ollama_ok = check_ollama_status()
    with active_requests_lock:
        req_count = active_requests
    with total_completed_lock:
        done_count = total_completed_jobs
    gpu_name, vram_gb = get_gpu_info()
    return {
        "status": "ok" if ollama_ok else "degraded",
        "service": "autosub-central-master-llm",
        "ollama_connected": ollama_ok,
        "ollama_url": OLLAMA_URL,
        "pipeline": f"2-Stage Cascade (Step 6 {STAGE1_MODEL} + Step 7 {STAGE2_MODEL})",
        "model": f"{STAGE1_MODEL} -> {STAGE2_MODEL}",
        "stage1_model": STAGE1_MODEL,
        "stage2_model": STAGE2_MODEL,
        "gpu": gpu_name,
        "role": "master_llm",
        "vram_gb": vram_gb,
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
            util = int(float(parts[0])) if parts[0].replace('.','',1).isdigit() else 0
            used_mb = int(float(parts[1])) if parts[1].replace('.','',1).isdigit() else 0
            total_mb = int(float(parts[2])) if parts[2].replace('.','',1).isdigit() else 32768
            temp = int(float(parts[3])) if parts[3].replace('.','',1).isdigit() else 0
            power = 0.0
            if len(parts) > 4:
                try:
                    power = round(float(parts[4]), 1)
                except Exception:
                    power = 0.0
            return {
                "gpu_util_percent": util,
                "mem_used_mb": used_mb,
                "mem_total_mb": total_mb,
                "mem_used_gb": round(used_mb / 1024, 2),
                "mem_total_gb": round(total_mb / 1024, 2),
                "temp_c": temp,
                "power_watts": power
            }
    except Exception:
        pass
    return {
        "gpu_util_percent": 0,
        "mem_used_mb": 0,
        "mem_total_mb": 32768,
        "mem_used_gb": 0,
        "mem_total_gb": 32.0,
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

def refine_subtitles_stage1_only(
    segments: List[Dict[str, Any]],
    drama_title: str = "ทั่วไป",
    known_chars: Optional[List[str]] = None,
    batch_size: int = 25,
    model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], float]:
    """
    Stage 1 Dedicated Endpoint (Step 6 Thai Colloquial Flow):
    Runs Typhoon 2.1 Gemma3 12B (or configured model) to refine drama spoken phrasing.
    """
    t_start = time.time()
    m_s1 = model or STAGE1_MODEL
    refined_segments = [dict(s) for s in segments]
    char_str = ", ".join(known_chars[:15]) if known_chars else "ไม่ระบุ"

    typhoon_system_prompt = (
        f"คุณคือ AI ผู้เชี่ยวชาญด้านภาษาไทยและบทละครโทรทัศน์ไทย (Thai Drama Colloquial Flow Refiner)\n"
        f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
        f"ตัวละครหลัก: {char_str}\n"
        "หน้าที่: ตรวจสอบและเกลาบทสนทนาภาษาไทยจากระบบเสียงพูด (ASR) ให้ถูกต้องตามสำนวนไทยและบริบทละคร\n"
        "1. แก้ไขคำเพี้ยน/คำพ้องเสียงและชื่อตัวละครตามบริบทละคร เช่น อาบดู -> แอบดู, ทามไม่ -> ทำไม, ชอบการ -> ชอบกัน, คะณชาติ -> คุณชาติ\n"
        "2. ห้ามแต่งเติมบทใหม่ รักษาความหมายเดิม\n"
        "3. ส่งผลลัพธ์เป็น JSON Object ที่มี key เป็น ID และ value เป็นประโยคที่เกลาแล้วทั้งหมด"
    )

    for i in range(0, len(refined_segments), batch_size):
        chunk = refined_segments[i:i + batch_size]
        orig_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}
        try:
            p_s1 = {
                "model": m_s1,
                "format": "json",
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": typhoon_system_prompt},
                    {"role": "user", "content": f"จงเกลาบทสนทนานี้ในรูปแบบ JSON:\n{json.dumps(orig_dict, ensure_ascii=False, indent=2)}"}
                ]
            }
            req_s1 = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps(p_s1, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
            with urllib.request.urlopen(req_s1, timeout=240.0) as resp1:
                r_json1 = json.loads(resp1.read().decode("utf-8"))
                cnt1 = r_json1.get("message", {}).get("content", "").strip()
                p1 = {}
                try:
                    p1 = json.loads(cnt1)
                except Exception:
                    m1 = re.search(r'\{.*\}', cnt1, re.DOTALL)
                    if m1:
                        p1 = json.loads(m1.group(0))
                if isinstance(p1, dict) and p1:
                    for idx, s in enumerate(chunk, start=i):
                        sid = str(s.get("id", idx))
                        if sid in p1 and isinstance(p1[sid], str) and p1[sid].strip():
                            s["text"] = p1[sid].strip()
        except Exception as err1:
            logger.warning(f"Batch {i // batch_size + 1} Stage 1 ({m_s1}) warning: {err1}. Keeping original cues.")
            continue

    dur = round(time.time() - t_start, 2)
    return refined_segments, dur

def refine_subtitles_stage2_only(
    segments: List[Dict[str, Any]],
    drama_title: str = "ทั่วไป",
    known_chars: Optional[List[str]] = None,
    batch_size: int = 25,
    model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int, float]:
    """
    Stage 2 Dedicated Endpoint (Step 7 Precision Subtitle QC & Diff Extraction):
    Runs Qwen 2.5 14B Q5_K_M to perform final subtitle proofreading and diff extraction.
    """
    t_start = time.time()
    m_s2 = model or STAGE2_MODEL
    refined_segments = [dict(s) for s in segments]
    total_corrected = 0
    char_str = ", ".join(known_chars[:15]) if known_chars else "ไม่ระบุ"

    qwen_system_prompt = (
        f"คุณคือ Subtitle QC และบรรณาธิการตรวจทานซับไตเติลขั้นสูง (Thai Subtitle Precision QC & Diff Extractor)\n"
        f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
        f"ตัวละครหลัก: {char_str}\n"
        "ภารกิจ: ตรวจสอบบทสนทนาภาษาไทยจากระบบเสียงพูด (ASR) และตรวจทานความถูกต้องขั้นสุดท้าย (Final Precision Check)\n"
        "กฎเหล็ก:\n"
        "1. ห้ามเติมคำแต่งเพิ่ม ห้ามเปลี่ยนโครงสร้างประโยค\n"
        "2. ตรวจสอบการสะกดคำ การใช้วรรณยุกต์ คะ/ค่ะ และคำพ้องเสียง\n"
        "3. ข้อห้ามเรื่องรูปแบบ: ส่งเฉพาะข้อความที่แก้ไขสมบูรณ์แล้วเท่านั้น ห้ามใส่เครื่องหมายลูกศร (-> หรือ →)\n"
        "4. ส่งผลลัพธ์เป็น Diff-Only JSON คืนเฉพาะ ID ที่มีการแก้ไขจากบทเดิมเท่านั้น เช่น {\"6\": \"ข้อความที่แก้แล้ว\"} หากไม่มีคำผิดเลยให้ส่ง {}"
    )

    for i in range(0, len(refined_segments), batch_size):
        chunk = refined_segments[i:i + batch_size]
        orig_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}
        try:
            user_prompt_s2 = (
                f"บทสนทนาต้นฉบับ:\n{json.dumps(orig_dict, ensure_ascii=False, indent=2)}\n\n"
                "ส่งผลลัพธ์เฉพาะ ID ที่ตรวจพบคำผิดและแก้ไขสมบูรณ์แล้วในรูปแบบ Diff JSON:"
            )
            p_s2 = {
                "model": m_s2,
                "format": "json",
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": qwen_system_prompt},
                    {"role": "user", "content": user_prompt_s2}
                ]
            }
            req_s2 = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps(p_s2, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
            with urllib.request.urlopen(req_s2, timeout=240.0) as resp2:
                r_json2 = json.loads(resp2.read().decode("utf-8"))
                cnt2 = r_json2.get("message", {}).get("content", "").strip()
                diff_map = {}
                try:
                    diff_map = json.loads(cnt2)
                except Exception:
                    m2 = re.search(r'\{.*\}', cnt2, re.DOTALL)
                    if m2:
                        diff_map = json.loads(m2.group(0))

                if isinstance(diff_map, dict):
                    for idx, s in enumerate(chunk, start=i):
                        sid = str(s.get("id", idx))
                        if sid in diff_map and isinstance(diff_map[sid], str) and diff_map[sid].strip():
                            new_text = diff_map[sid].strip()
                            if "->" in new_text or "→" in new_text:
                                parts = re.split(r'\s*(?:->|→)\s*', new_text)
                                new_text = parts[-1].strip()
                            new_text = re.sub(r'^[\'"]|[\'"]$', '', new_text).strip()
                            if new_text and new_text != s.get("text", ""):
                                s["text"] = new_text
                                total_corrected += 1
        except Exception as err2:
            logger.warning(f"Batch {i // batch_size + 1} Stage 2 ({m_s2}) warning: {err2}. Retaining cues.")
            continue

    dur = round(time.time() - t_start, 2)
    return refined_segments, total_corrected, dur

_CACHED_STAGE1_URL = None
_CACHED_STAGE1_TIME = 0.0

def get_stage1_llm_url() -> str:
    """Resolve Tier 2 (Step 6) LLM endpoint URL from env or Storage Server."""
    global _CACHED_STAGE1_URL, _CACHED_STAGE1_TIME
    now = time.time()
    if _CACHED_STAGE1_URL and (now - _CACHED_STAGE1_TIME < 60.0):
        return _CACHED_STAGE1_URL

    env_url = os.environ.get("STAGE1_LLM_URL", "").strip().rstrip("/")
    if env_url:
        _CACHED_STAGE1_URL = env_url
        _CACHED_STAGE1_TIME = now
        return env_url

    storage_candidates = [
        "http://ph.cdnwatch.com/subtitle.php?action=get_master_llm",
        "http://23.158.40.152/subtitle.php?action=get_master_llm"
    ]
    for surl in storage_candidates:
        try:
            req = urllib.request.Request(surl, headers={"User-Agent": "AutoSub-Tier3-Probe"})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    s1 = data.get("stage1_llm_url", "").strip().rstrip("/")
                    if s1:
                        _CACHED_STAGE1_URL = s1
                        _CACHED_STAGE1_TIME = now
                        return s1
        except Exception:
            pass

    return ""

def refine_subtitles_cascade(
    segments: List[Dict[str, Any]],
    drama_title: str = "ทั่วไป",
    known_chars: Optional[List[str]] = None,
    batch_size: int = 25,
    stage1_model: Optional[str] = None,
    stage2_model: Optional[str] = None
) -> Tuple[List[Dict[str, Any]], int, float]:
    """
    2-Stage Cascade Subtitle Refinement Pipeline ($0 Cost / Decoupled GPU Cluster):
      Stage 1 (Step 6): Typhoon 2.1 Gemma3 12B / Llama 3.1 Typhoon 8B
        - Specializes in Thai spoken flow, colloquial drama phrasing, and cultural context.
        - Delegated to Tier 2 Node (RTX 3060) if STAGE1_LLM_URL is set, or runs locally.
      Stage 2 (Step 7): Qwen 2.5 14B Q5_K_M
        - Specializes in strict Subtitle QC, particle precision (คะ/ค่ะ), and Diff-Only extraction.
        - Compares raw cues with Stage 1 output and returns only verified corrections as JSON diff.
    Fallback:
      - If Stage 1 fails, falls back directly to single-pass Stage 2.
      - If Stage 2 fails, keeps original Step 5 cues to guarantee zero subtitle loss.
    """
    t_start = time.time()
    m_s1 = stage1_model or STAGE1_MODEL
    m_s2 = stage2_model or STAGE2_MODEL
    refined_segments = [dict(s) for s in segments]
    total_corrected = 0
    char_str = ", ".join(known_chars[:15]) if known_chars else "ไม่ระบุ"

    typhoon_system_prompt = (
        f"คุณคือ AI ผู้เชี่ยวชาญด้านภาษาไทยและบทละครโทรทัศน์ไทย (Thai Drama Colloquial Flow Refiner)\n"
        f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
        f"ตัวละครหลัก: {char_str}\n"
        "หน้าที่: ตรวจสอบและเกลาบทสนทนาภาษาไทยจากระบบเสียงพูด (ASR) ให้ถูกต้องตามสำนวนไทยและบริบทละคร\n"
        "1. แก้ไขคำเพี้ยน/คำพ้องเสียงและชื่อตัวละครตามบริบทละคร เช่น อาบดู -> แอบดู, ทามไม่ -> ทำไม, ชอบการ -> ชอบกัน, คะณชาติ -> คุณชาติ\n"
        "2. ห้ามแต่งเติมบทใหม่ รักษาความหมายเดิม\n"
        "3. ส่งผลลัพธ์เป็น JSON Object ที่มี key เป็น ID และ value เป็นประโยคที่เกลาแล้วทั้งหมด"
    )

    qwen_system_prompt = (
        f"คุณคือ Subtitle QC และบรรณาธิการตรวจทานซับไตเติลขั้นสูง (Thai Subtitle Precision QC & Diff Extractor)\n"
        f"ข้อมูลละคร: เรื่อง '{drama_title}'\n"
        f"ตัวละครหลัก: {char_str}\n"
        "ภารกิจ: รับบทสนทนาต้นฉบับและบทสนทนาที่ผ่านการเกลาบริบทมาแล้ว ตรวจทานความถูกต้องขั้นสุดท้าย (Final Precision Check)\n"
        "กฎเหล็ก:\n"
        "1. ห้ามเติมคำแต่งเพิ่ม ห้ามเปลี่ยนโครงสร้างประโยค\n"
        "2. ตรวจสอบการสะกดคำ การใช้วรรณยุกต์ คะ/ค่ะ และคำพ้องเสียง\n"
        "3. ข้อห้ามเรื่องรูปแบบ: ส่งเฉพาะข้อความที่แก้ไขสมบูรณ์แล้วเท่านั้น ห้ามใส่เครื่องหมายลูกศร (-> หรือ →)\n"
        "4. ส่งผลลัพธ์เป็น Diff-Only JSON คืนเฉพาะ ID ที่มีการแก้ไขจากบทเดิมเท่านั้น เช่น {\"6\": \"ข้อความที่แก้แล้ว\"} หากไม่มีคำผิดเลยให้ส่ง {}"
    )

    s1_remote_url = get_stage1_llm_url()

    for i in range(0, len(refined_segments), batch_size):
        chunk = refined_segments[i:i + batch_size]
        orig_dict = {str(s.get("id", idx)): s.get("text", "") for idx, s in enumerate(chunk, start=i)}

        # Stage 1: Typhoon (Thai Spoken Context & Idiom Flow)
        stage1_dict = orig_dict
        if s1_remote_url and "127.0.0.1" not in s1_remote_url and "localhost" not in s1_remote_url:
            try:
                sub_payload = {
                    "segments": [{"id": k, "text": v} for k, v in orig_dict.items()],
                    "drama_title": drama_title,
                    "known_chars": known_chars,
                    "batch_size": batch_size,
                    "stage1_model": m_s1
                }
                req_s1 = urllib.request.Request(
                    f"{s1_remote_url}/v1/llm/stage1",
                    data=json.dumps(sub_payload, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )
                with urllib.request.urlopen(req_s1, timeout=60.0) as resp1:
                    r_json1 = json.loads(resp1.read().decode("utf-8"))
                    if r_json1.get("status") == "success" and r_json1.get("segments"):
                        stage1_dict = {str(s.get("id")): s.get("text") for s in r_json1.get("segments", [])}
            except Exception as err1:
                logger.warning(f"Batch {i // batch_size + 1} Remote Stage 1 ({s1_remote_url}) warning: {err1}. Falling back directly to Stage 2.")
                stage1_dict = orig_dict
        else:
            try:
                p_s1 = {
                    "model": m_s1,
                    "format": "json",
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
                    "messages": [
                        {"role": "system", "content": typhoon_system_prompt},
                        {"role": "user", "content": f"จงเกลาบทสนทนานี้ในรูปแบบ JSON:\n{json.dumps(orig_dict, ensure_ascii=False, indent=2)}"}
                    ]
                }
                req_s1 = urllib.request.Request(
                    f"{OLLAMA_URL}/api/chat",
                    data=json.dumps(p_s1, ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type": "application/json; charset=utf-8"}
                )
                with urllib.request.urlopen(req_s1, timeout=240.0) as resp1:
                    r_json1 = json.loads(resp1.read().decode("utf-8"))
                    cnt1 = r_json1.get("message", {}).get("content", "").strip()
                    try:
                        p1 = json.loads(cnt1)
                        if isinstance(p1, dict) and p1:
                            stage1_dict = {**orig_dict, **p1}
                    except Exception:
                        m1 = re.search(r'\{.*\}', cnt1, re.DOTALL)
                        if m1:
                            p1 = json.loads(m1.group(0))
                            if isinstance(p1, dict) and p1:
                                stage1_dict = {**orig_dict, **p1}
            except Exception as err1:
                logger.warning(f"Batch {i // batch_size + 1} Local Stage 1 ({m_s1}) warning: {err1}. Falling back directly to Stage 2.")
                stage1_dict = orig_dict

        # Stage 2: Qwen 2.5 14B Q6_K (Diff-Only Precision Subtitle QC)
        try:
            user_prompt_s2 = (
                f"เทียบต้นฉบับเดิม:\n{json.dumps(orig_dict, ensure_ascii=False, indent=2)}\n\n"
                f"กับฉบับที่ผ่านการเกลาบริบท:\n{json.dumps(stage1_dict, ensure_ascii=False, indent=2)}\n\n"
                "ส่งผลลัพธ์เฉพาะ ID ที่ถูกต้องสมบูรณ์และต่างจากต้นฉบับเดิมในรูปแบบ JSON:"
            )
            p_s2 = {
                "model": m_s2,
                "format": "json",
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.1, "num_predict": 512, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": qwen_system_prompt},
                    {"role": "user", "content": user_prompt_s2}
                ]
            }
            req_s2 = urllib.request.Request(
                f"{OLLAMA_URL}/api/chat",
                data=json.dumps(p_s2, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json; charset=utf-8"}
            )
            with urllib.request.urlopen(req_s2, timeout=240.0) as resp2:
                r_json2 = json.loads(resp2.read().decode("utf-8"))
                cnt2 = r_json2.get("message", {}).get("content", "").strip()
                diff_map = {}
                try:
                    diff_map = json.loads(cnt2)
                except Exception:
                    m2 = re.search(r'\{.*\}', cnt2, re.DOTALL)
                    if m2:
                        diff_map = json.loads(m2.group(0))

                if isinstance(diff_map, dict):
                    for idx, s in enumerate(chunk, start=i):
                        sid = str(s.get("id", idx))
                        if sid in diff_map and isinstance(diff_map[sid], str) and diff_map[sid].strip():
                            new_text = diff_map[sid].strip()
                            if "->" in new_text or "→" in new_text:
                                parts = re.split(r'\s*(?:->|→)\s*', new_text)
                                new_text = parts[-1].strip()
                            new_text = re.sub(r'^[\'"]|[\'"]$', '', new_text).strip()
                            if new_text and new_text != s.get("text", ""):
                                s["text"] = new_text
                                total_corrected += 1
        except Exception as err2:
            logger.warning(f"Batch {i // batch_size + 1} Stage 2 ({m_s2}) warning: {err2}. Retaining cues.")
            continue

    dur = round(time.time() - t_start, 2)
    return refined_segments, total_corrected, dur

# ==============================================================================
# 1. CONTEXTUAL THAI PROOFREADING (Step 6 & 7: 2-Stage Cascade / Modular Endpoints)
# ==============================================================================
@app.post("/v1/llm/stage1")
def refine_stage1_endpoint(req_data: RefineRequest):
    """Modular Endpoint: Executes ONLY Stage 1 (Typhoon 2.1 Gemma3 12B Thai Colloquial Flow)."""
    segments = req_data.segments
    if not segments:
        return {"status": "success", "segments": [], "duration_sec": 0.0}
    drama_title = req_data.drama_title or "ทั่วไป"
    known_chars = req_data.known_chars or []
    batch_size = req_data.batch_size or 25
    m_s1 = req_data.stage1_model or STAGE1_MODEL

    refined_segments, dur = refine_subtitles_stage1_only(
        segments=segments,
        drama_title=drama_title,
        known_chars=known_chars,
        batch_size=batch_size,
        model=m_s1
    )
    return {
        "status": "success",
        "stage": 1,
        "model": m_s1,
        "segments": refined_segments,
        "duration_sec": dur
    }

@app.post("/v1/llm/stage2")
def refine_stage2_endpoint(req_data: RefineRequest):
    """Modular Endpoint: Executes ONLY Stage 2 (Qwen 2.5 14B Q5_K_M Precision Subtitle QC & Diff Extractor)."""
    segments = req_data.segments
    if not segments:
        return {"status": "success", "corrected_count": 0, "segments": [], "duration_sec": 0.0}
    drama_title = req_data.drama_title or "ทั่วไป"
    known_chars = req_data.known_chars or []
    batch_size = req_data.batch_size or int(os.environ.get("BATCH_SIZE", "25"))
    m_s2 = req_data.stage2_model or STAGE2_MODEL

    refined_segments, total_corrected, dur = refine_subtitles_stage2_only(
        segments=segments,
        drama_title=drama_title,
        known_chars=known_chars,
        batch_size=batch_size,
        model=m_s2
    )
    return {
        "status": "success",
        "stage": 2,
        "model": m_s2,
        "corrected_count": total_corrected,
        "segments": refined_segments,
        "duration_sec": dur
    }

@app.post("/v1/llm/refine")
def refine_subtitles(req_data: RefineRequest):
    global active_requests, total_completed_jobs
    segments = req_data.segments
    if not segments:
        return {"status": "success", "corrected_count": 0, "segments": [], "duration_sec": 0.0}

    drama_title = req_data.drama_title or "ทั่วไป"
    known_chars = req_data.known_chars or []
    batch_size = int(os.environ.get("BATCH_SIZE", "25"))
    if req_data.batch_size and 5 <= req_data.batch_size <= 25:
        batch_size = req_data.batch_size
    else:
        batch_size = min(batch_size, 25)

    if not check_ollama_status():
        logger.warning(f"Ollama is unreachable at {OLLAMA_URL}. Returning original cues as fallback.")
        return {
            "status": "fallback",
            "message": "Ollama service unavailable on Master Node",
            "corrected_count": 0,
            "segments": segments,
            "duration_sec": 0.0
        }

    acquired = LLM_SEMAPHORE.acquire(timeout=1800.0)
    if not acquired:
        logger.warning(f"Master LLM semaphore timeout after 1800s for drama '{drama_title}'. Returning original cues.")
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
    m_s1 = req_data.stage1_model or STAGE1_MODEL
    m_s2 = req_data.stage2_model or STAGE2_MODEL
    node_role = os.environ.get("ROLE", "cascade")

    try:
        if node_role in ["step7", "stage2"]:
            logger.info(f"Processing Dedicated Stage 2 QC: {len(segments)} cues ({m_s2}), drama='{drama_title}'")
            refined_segments, total_corrected, dur = refine_subtitles_stage2_only(
                segments=segments,
                drama_title=drama_title,
                known_chars=known_chars,
                batch_size=batch_size,
                model=m_s2
            )
            pipeline_desc = f"Tier 3 Dedicated QC ({m_s2})"
        elif node_role in ["step6", "stage1"]:
            logger.info(f"Processing Dedicated Stage 1 Flow: {len(segments)} cues ({m_s1}), drama='{drama_title}'")
            refined_segments, dur = refine_subtitles_stage1_only(
                segments=segments,
                drama_title=drama_title,
                known_chars=known_chars,
                batch_size=batch_size,
                model=m_s1
            )
            total_corrected = sum(1 for o, r in zip(segments, refined_segments) if o.get("text") != r.get("text"))
            pipeline_desc = f"Tier 2 Dedicated Flow ({m_s1})"
        else:
            logger.info(f"Processing 2-Stage Cascade refinement: {len(segments)} cues ({m_s1} -> {m_s2}), drama='{drama_title}', chars={len(known_chars)}")
            refined_segments, total_corrected, dur = refine_subtitles_cascade(
                segments=segments,
                drama_title=drama_title,
                known_chars=known_chars,
                batch_size=batch_size,
                stage1_model=m_s1,
                stage2_model=m_s2
            )
            pipeline_desc = f"2-Stage Cascade ({m_s1} + {m_s2})"

        with total_completed_lock:
            total_completed_jobs += 1
        logger.info(f"Refinement completed: {total_corrected} cues corrected in {dur}s for drama '{drama_title}'.")
        return {
            "status": "success",
            "pipeline": pipeline_desc,
            "stage1_model": m_s1,
            "stage2_model": m_s2,
            "corrected_count": total_corrected,
            "segments": refined_segments,
            "duration_sec": dur
        }
    except Exception as ex:
        dur = round(time.time() - t_start, 2)
        logger.error(f"Error during 2-Stage Cascade refinement: {ex}")
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
    logger.info(f"[ASYNC START] Processing 2-Stage Cascade refinement for custom_id='{custom_id}', drama='{job.drama_title}', cues={len(job.segments)}")
    
    with active_requests_lock:
        active_requests += 1

    # Generous queue timeout (6.0 hours / 21600s)
    acquired = LLM_SEMAPHORE.acquire(timeout=21600.0)
    if not acquired:
        logger.error(f"[ASYNC TIMEOUT] Semaphore queue timeout (21600s / 6h) for custom_id='{custom_id}'")
        with active_requests_lock:
            active_requests -= 1
        _send_fallback_webhook(job, "LLM queue busy")
        return

    logger.info(f"[ASYNC QUEUE ACQUIRED] Semaphore slot acquired for custom_id='{custom_id}', drama='{job.drama_title}' after {round(time.time() - t_start, 1)}s queue wait.")

    try:
        segments = [dict(s) for s in job.segments]
        drama_title = job.drama_title or "ทั่วไป"
        known_chars = job.known_chars or []
        batch_size = int(os.environ.get("BATCH_SIZE", "25"))
        if job.batch_size and 5 <= job.batch_size <= 25:
            batch_size = job.batch_size
        else:
            batch_size = min(batch_size, 25)

        refined_segments, total_corrected, dur = refine_subtitles_cascade(
            segments=segments,
            drama_title=drama_title,
            known_chars=known_chars,
            batch_size=batch_size
        )

        total_proc = round((job.processing_time_worker or 0.0) + dur, 2)
        logger.info(f"[ASYNC SUCCESS] Refined {total_corrected} cues in {dur}s for custom_id='{custom_id}' (Total time: {total_proc}s). Delivering webhook...")

        # Format VTT
        vtt_lines = ["WEBVTT\n"]
        for s in refined_segments:
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
        for i_idx, s in enumerate(refined_segments, 1):
            srt_lines.append(str(i_idx))
            srt_lines.append(f"{format_timestamp(s['start'], vtt=False)} --> {format_timestamp(s['end'], vtt=False)}")
            spk = s.get("speaker")
            txt = format_thai_subtitle_line_wrap(s.get("text", ""))
            if spk:
                srt_lines.append(f"[{spk}]: {txt}\n")
            else:
                srt_lines.append(f"{txt}\n")
        srt_out = "\n".join(srt_lines)
        combined_text = " ".join(s.get("text", "") for s in refined_segments)

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
            "segments": refined_segments,
            "step6_refined": True,
            "corrected_count": total_corrected,
            "pipeline": "2-Stage Cascade (Typhoon 2 8B + Qwen 14B Q8)"
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
    # Master Node runs DEFAULT_MODEL (e.g. qwen2.5:32b). Auto-map any incoming model name.
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
            with urllib.request.urlopen(req, timeout=240.0) as resp:
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
                with urllib.request.urlopen(req, timeout=240.0) as resp:
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
