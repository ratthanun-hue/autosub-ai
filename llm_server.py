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
import logging
import threading
import subprocess
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
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

# Concurrency control: Allow at most 2 concurrent batch refinements to protect VRAM and latency
LLM_SEMAPHORE = threading.Semaphore(2)
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

    acquired = LLM_SEMAPHORE.acquire(timeout=60.0)
    if not acquired:
        logger.warning(f"Master LLM semaphore timeout after 60s for drama '{drama_title}'. Returning original cues.")
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
            "1. แก้ไขชื่อตัวละครที่ฟังเพี้ยนให้ตรงกับรายชื่อตัวละครหลัก เช่น หากได้ยินชื่อเพี้ยนหรือพ้องเสียงใกล้เคียง (เช่น พี่ดนทร์/พี่ผู้ชม/พี่โดน -> พี่ดล หรือ คุณภูดล, นี้ซิริ/เนซีรี -> เนตรศิริ) ให้แก้เป็นชื่อตัวละครที่ถูกต้องตามบริบท\n"
            "2. แก้ไขคำพ้องเสียงหรือคำที่ Whisper ฟังเพี้ยนจากเสียงพูด เช่น:\n"
            "   - เสียงพยัญชนะ/สระเพี้ยน: 'เต้นความ' -> 'แจ้งความ', 'จุดรวช' -> 'ตำรวจ', 'พลิกแฟร์ม' -> 'พลิกแฟ้ม', 'แล่ว' -> 'แล้ว', 'เหน' -> 'เห็น'\n"
            "   - คำไม่มีความหมายหรือผิดไวยากรณ์: 'โมโมง/มองหมอก' -> 'หมองมัว', 'สาปศูนย์' -> 'สาบสูญ'\n"
            "3. ข้อห้ามเด็ดขาด: ห้ามแต่งประโยคใหม่, ห้ามเติมคำลงท้าย (เช่น ห้ามเติม ค่ะ/ครับ/ฮะ ถ้าต้นฉบับไม่มี), และห้ามตัดทอนคำออก\n"
            "4. ส่งผลลัพธ์กลับมาเป็น JSON Object ตาม ID เดิมเป๊ะ ในรูปแบบ {\"ID\": \"ข้อความ\"}"
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
                        "num_predict": 2048
                    },
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"จงตรวจแก้ซับไตเติลต่อไปนี้:\n{json.dumps(cues_dict, ensure_ascii=False, indent=2)}"}
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
