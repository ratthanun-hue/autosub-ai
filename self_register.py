#!/usr/bin/env python3
import time
import json
import os
import urllib.request
import urllib.parse

VAST_API_KEY = os.environ.get("VAST_API_KEY", "bb158182f28dba3c4d30c71fd31eca1149c65b308b7f59ead54c0a5c66332a5d")
STORAGE_REGISTER_URL = os.environ.get("STORAGE_REGISTER_URL", "http://ph.cdnwatch.com/subtitle.php?action=register_node")
LOG_FILE = "/root/whisper-server/self_register.log"

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}"
    print(line)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass

log("=== STARTING NODE SELF-REGISTRATION ===")

label_file = "/root/.vast_containerlabel"
inst_id = None
if os.path.exists(label_file):
    try:
        inst_id = open(label_file).read().strip().replace("C.", "")
    except Exception as e:
        log(f"Error reading container label: {e}")

if not inst_id:
    # Fallback to hostname or env
    inst_id = os.environ.get("VAST_CONTAINERLABEL", "").replace("C.", "") or os.uname().nodename

log(f"Detected Instance ID: {inst_id}")

# 1. Wait for local whisper service to respond on http://localhost:10100/health
log("Step 1: Waiting for local Whisper service to be ready on port 10100...")
local_ready = False
for attempt in range(40):
    try:
        with urllib.request.urlopen("http://localhost:10100/health", timeout=3) as r:
            if r.status == 200:
                local_ready = True
                log("Local Whisper service is READY!")
                break
    except Exception:
        pass
    time.sleep(3)

if not local_ready:
    log("WARNING: Local service did not become healthy within 120s, proceeding anyway...")

# 2. Query Vast.ai API to discover external IP and mapped port for 10100/tcp
log("Step 2: Discovering external IP and port mappings from Vast.ai API...")
public_ip = None
whisper_port = None
gpu_name = "Tesla V100"

for attempt in range(30):
    try:
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
            gpu_name = target_inst.get("gpu_name", "Tesla V100")
            
            if "10100/tcp" in ports and len(ports["10100/tcp"]) > 0:
                whisper_port = ports["10100/tcp"][0].get("HostPort")
            
            if public_ip and whisper_port:
                log(f"Discovered Network: IP={public_ip}, Port={whisper_port}, GPU={gpu_name}")
                break
            else:
                log(f"Attempt {attempt+1}: IP={public_ip}, Port={whisper_port}, waiting 5s for port assignment...")
        else:
            log(f"Attempt {attempt+1}: Instance {inst_id} not listed in Vast API yet...")
    except Exception as e:
        log(f"Attempt {attempt+1}: Vast API query failed: {e}")
    
    time.sleep(5)

if not public_ip or not whisper_port:
    log("FAILED: Could not discover external IP/Port within timeout.")
    exit(1)

# 3. Post registration payload to Storage Server
log(f"Step 3: Registering node with Storage Server at {STORAGE_REGISTER_URL}...")
payload = {
    "ip": public_ip,
    "port": str(whisper_port),
    "id": str(inst_id),
    "name": f"Dynamic GPU #{inst_id}",
    "gpu": gpu_name
}

post_data = urllib.parse.urlencode(payload).encode("utf-8")
req_reg = urllib.request.Request(STORAGE_REGISTER_URL, data=post_data, method="POST")

try:
    with urllib.request.urlopen(req_reg, timeout=10) as r:
        resp_text = r.read().decode()
        log(f"Registration Response: HTTP {r.status} - {resp_text}")
        log("=== REGISTRATION COMPLETE: NODE IS NOW RECEIVING JOBS! ===")
except Exception as e:
    log(f"Registration FAILED: {e}")
    exit(1)
