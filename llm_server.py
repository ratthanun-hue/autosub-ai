#!/usr/bin/env python3
"""
AutoSub-AI Central Master LLM Server (Step 6 Proofreader)
Runs on Master Node 1 (Tesla V100 32GB) on port 10200 (Vast.ai mapped port 40316).
Provides high-speed contextual Thai subtitle proofreading via local Ollama (Qwen2.5-7B).
Delegated to by Dynamic Worker Nodes when local Ollama is not present.
"""

import os
import re
import time
import json
import logging
import threading
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Logging setup
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
    description="Contextual Thai Subtitle Refinement using Qwen2.5-7B on Master Node",
    version="1.0.0"
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")

# Concurrency control: Allow at most 2 concurrent batch refinements to protect VRAM and latency
LLM_SEMAPHORE = threading.Semaphore(2)
active_requests = 0
active_requests_lock = threading.Lock()

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

def check_ollama_status() -> bool:
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", headers={"User-Agent": "Master-LLM-Probe"})
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            return resp.status == 200
    except Exception:
        return False

@app.get("/health")
def health_check():
    ollama_ok = check_ollama_status()
    with active_requests_lock:
        req_count = active_requests
    return {
        "status": "ok" if ollama_ok else "degraded",
        "service": "autosub-central-master-llm",
        "ollama_connected": ollama_ok,
        "ollama_url": OLLAMA_URL,
        "model": DEFAULT_MODEL,
        "active_jobs": req_count,
        "server_time": time.time()
    }

@app.post("/v1/llm/refine")
def refine_subtitles(req_data: RefineRequest):
    global active_requests
    segments = req_data.segments
    if not segments:
        return {"status": "success", "corrected_count": 0, "segments": [], "duration_sec": 0.0}

    drama_title = req_data.drama_title or "ทั่วไป"
    known_chars = req_data.known_chars or []
    model = req_data.model or DEFAULT_MODEL
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
            "   - เสียงพยัญชนะ/สระเพี้ยน: 'เต้นความ' -> 'แจ้งความ', 'จุดรวช' -> 'ตำรวจ', 'พลิกแฟร์ม' -> 'พลิกแฟ้ม'\n"
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

                with urllib.request.urlopen(req, timeout=45.0) as resp:
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("LLM_PORT", 10200))
    logger.info(f"Starting AutoSub-AI Master LLM Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
