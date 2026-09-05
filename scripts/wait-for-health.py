#!/usr/bin/env python3
"""Wait for a server health endpoint without using configured proxies."""
import argparse
import json
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000/health")
parser.add_argument("--timeout", type=float, default=600)
args = parser.parse_args()
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + args.timeout
while time.monotonic() < deadline:
    try:
        with opener.open(args.url, timeout=5) as response:
            if json.load(response).get("status") == "ok":
                print("ready")
                raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit("timed out waiting for model health")

