#!/usr/bin/env python3
"""
AutoSub-AI Master Node Self-Registration & Heartbeat Daemon
Runs on Master LLM Node (container port 10200) to:
1. Wait for local Ollama (11434) and Master LLM Service (10200) to be fully ready.
2. Discover public IP and mapped host port for 10200/tcp via Vast.ai API.
3. Register with Storage Server (ph.cdnwatch.com/subtitle.php?action=register_master_llm).
4. Send heartbeat ping every 30s to keep registration fresh.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.parse
import urllib.error

VAST_API_KEY = os.environ.get("VAST_API_KEY", "")
if not VAST_API_KEY:
    for kp in ["/root/.vast_api_key", "/root/.config/vastai/vast_api_key"]:
        if os.path.exists(kp):
            try:
                content = open(kp).read().strip()
                if content:
                    VAST_API_KEY = content
                    break
            except Exception:
                pass

STORAGE_REGISTER_URL = os.environ.get(
    "STORAGE_MASTER_REGISTER_URL",
    "http://ph.cdnwatch.com/subtitle.php?action=register_master_llm"
)
MASTER_AUTH_TOKEN = os.environ.get("MASTER_AUTH_TOKEN", "autosub_master_secret_2026")
LOG_FILE = "/root/whisper-server/self_register_master.log"

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] [Master-Register] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

log("=== STARTING MASTER LLM SELF-REGISTRATION ===")

# Detect Instance ID
label_file = "/root/.vast_containerlabel"
inst_id = None
if os.path.exists(label_file):
    try:
        inst_id = open(label_file).read().strip().replace("C.", "")
    except Exception as e:
        log(f"Error reading container label: {e}")

if not inst_id:
    inst_id = (
        os.environ.get("CONTAINER_ID")
        or os.environ.get("VAST_CONTAINERLABEL", "").replace("C.", "")
        or os.uname().nodename
    )

log(f"Detected Master Instance ID: {inst_id}")

# 1. Wait for local Ollama & Master LLM service on port 10200
log("Step 1: Waiting for local Master LLM service to be ready on port 10200...")
local_ready = False
for attempt in range(60):  # up to 3 minutes
    try:
        with urllib.request.urlopen("http://localhost:10200/health", timeout=3) as r:
            if r.status == 200:
                local_ready = True
                log("Local Master LLM service (port 10200) is READY!")
                break
    except Exception:
        pass
    time.sleep(3)

if not local_ready:
    log("WARNING: Local Master LLM did not respond within 180s, proceeding to network discovery...")

# 2. Discover Public IP and Mapped Port for 10200/tcp
log("Step 2: Discovering external IP and port mappings from Vast.ai API...")
public_ip = os.environ.get("PUBLIC_IP")
master_port = os.environ.get("MASTER_PORT")
gpu_name = "NVIDIA RTX A4000"

if not public_ip or not master_port:
    for attempt in range(30):
        try:
            if not VAST_API_KEY:
                log("No VAST_API_KEY found, check environment or /root/.vast_api_key")
                break
            url = "https://console.vast.ai/api/v1/instances/"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {VAST_API_KEY}"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            target_inst = None
            for item in data.get("instances", []):
                if str(item.get("id")) == str(inst_id):
                    target_inst = item
                    break

            if target_inst:
                public_ip = target_inst.get("public_ipaddr")
                ports = target_inst.get("ports", {})
                gpu_name = target_inst.get("gpu_name", "NVIDIA RTX A4000")

                # Look for mapped 10200/tcp
                if "10200/tcp" in ports and len(ports["10200/tcp"]) > 0:
                    master_port = ports["10200/tcp"][0].get("HostPort")
                elif "10100/tcp" in ports and len(ports["10100/tcp"]) > 0:
                    # Fallback if unified single port
                    master_port = ports["10100/tcp"][0].get("HostPort")

                if public_ip and master_port:
                    log(f"Discovered Network: IP={public_ip}, Port={master_port}, GPU={gpu_name}")
                    break
                else:
                    log(f"Attempt {attempt+1}: IP={public_ip}, Port={master_port}, waiting 5s for port assignment...")
            else:
                log(f"Attempt {attempt+1}: Instance {inst_id} not listed in Vast API yet...")
        except Exception as e:
            log(f"Attempt {attempt+1}: Vast API query failed: {e}")

        time.sleep(5)

if not public_ip or not master_port:
    log("FAILED: Could not discover external IP/Port within timeout.")
    # If failed to query vast, check if we can register local IP or exit
    sys.exit(1)

# 3. Post registration payload to Storage Server
def send_registration():
    try:
        payload = {
            "ip": public_ip,
            "port": str(master_port),
            "id": str(inst_id),
            "model": "qwen2.5:14b",
            "token": MASTER_AUTH_TOKEN
        }
        post_data = urllib.parse.urlencode(payload).encode("utf-8")
        req_reg = urllib.request.Request(STORAGE_REGISTER_URL, data=post_data, method="POST")
        with urllib.request.urlopen(req_reg, timeout=10) as resp:
            resp_body = resp.read().decode()
            log(f"Registration response: {resp.status} - {resp_body}")
            return resp.status == 200
    except Exception as e:
        log(f"Registration error: {e}")
        return False

log(f"Step 3: Registering Master Node at {STORAGE_REGISTER_URL}...")
registered = send_registration()

# 4. Continuous Heartbeat Loop
log("Step 4: Entering Heartbeat loop (every 30s)...")
while True:
    time.sleep(30)
    try:
        # Check if local Master service is still alive
        is_healthy = False
        try:
            with urllib.request.urlopen("http://localhost:10200/health", timeout=3) as r:
                is_healthy = (r.status == 200)
        except Exception:
            is_healthy = False

        if is_healthy:
            send_registration()
        else:
            log("WARN: Local Master LLM on 10200 not responding, skipping heartbeat this cycle")
    except Exception as e:
        log(f"Heartbeat loop exception: {e}")
