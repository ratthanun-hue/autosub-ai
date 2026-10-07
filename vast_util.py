import sys, json, urllib2

cfg = json.loads(open("/usr/share/nginx/html/cache/vast_config.json").read())
api_key = cfg.get("api_key") or cfg.get("vast_api_key")

cmd = sys.argv[1] if len(sys.argv) > 1 else "list"

if cmd == "list":
    req = urllib2.Request("https://console.vast.ai/api/v1/instances/")
    req.add_header("Authorization", "Bearer " + api_key)
    resp = json.loads(urllib2.urlopen(req).read())
    for inst in resp.get("instances", []):
        print("ID: %s | Status: %s | Ports: %s | GPU: %s" % (
            inst.get("id"), inst.get("actual_status"), inst.get("ports"), inst.get("gpu_name")
        ))
elif cmd == "delete":
    target_id = sys.argv[2]
    req = urllib2.Request("https://console.vast.ai/api/v0/instances/%s/" % target_id)
    req.get_method = lambda: "DELETE"
    req.add_header("Authorization", "Bearer " + api_key)
    resp = urllib2.urlopen(req).read()
    print("Deleted %s: %s" % (target_id, resp))
