#!/usr/bin/env python3
"""
Local sample viewer — runs on your machine.
Periodically pulls sample JSON from the cluster via Heimdall proxy.
Auto-detects the latest samples_*.json file, or switch runs in the UI.

Usage:
    python eval/sample_viewer.py --proxy http://10.0.0.19:2222 --remote-dir /home/athuser/luxi-files/molloy
"""

import argparse
import json
import threading
import time
import urllib.request
import urllib.error

from flask import Flask, jsonify, render_template_string, request as flask_request

app = Flask(__name__)

SAMPLES_DATA = {"prompts": [], "samples": {}}
PROXY_URL = ""
REMOTE_DIR = ""
POLL_INTERVAL = 15
LAST_FETCH = "never"
FETCH_ERROR = ""
CURRENT_FILE = ""
AVAILABLE_FILES = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample Tracker</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Georgia', serif; background: #faf9f6; color: #2a2a2a; padding: 24px; }
        h1 { font-size: 1.4em; font-weight: normal; margin-bottom: 4px; color: #555; }
        .meta { font-size: 0.85em; color: #888; margin-bottom: 20px; }
        .run-select { margin-bottom: 16px; }
        .run-select select { font-family: 'Georgia', serif; font-size: 0.9em; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; background: white; }
        .run-select label { font-size: 0.85em; color: #888; margin-right: 6px; }
        .prompt-section { margin-bottom: 32px; border-bottom: 1px solid #e0ddd8; padding-bottom: 24px; }
        .prompt-text {
            font-size: 1.05em; font-style: italic; color: #444;
            margin-bottom: 12px; padding: 8px 12px;
            background: #f0eeea; border-radius: 4px;
            white-space: pre-wrap; line-height: 1.5;
        }
        .steps-row { display: flex; gap: 12px; overflow-x: auto; padding-bottom: 8px; }
        .step-card { min-width: 320px; max-width: 400px; flex-shrink: 0; background: white; border: 1px solid #e0ddd8; border-radius: 6px; padding: 12px 16px; }
        .step-label { font-size: 0.8em; font-weight: bold; color: #888; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
        .completion { font-size: 0.92em; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }
        .controls { position: fixed; top: 12px; right: 24px; display: flex; gap: 8px; align-items: center; }
        .controls button { padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px; background: white; cursor: pointer; font-size: 0.85em; }
        .controls button:hover { background: #f0f0f0; }
        .status { font-size: 0.8em; color: #888; }
        .no-data { color: #aaa; font-style: italic; }
        .prompt-nav { display: flex; gap: 6px; margin-bottom: 16px; flex-wrap: wrap; }
        .prompt-nav a { font-size: 0.8em; padding: 4px 10px; background: #eee; border-radius: 3px; text-decoration: none; color: #555; }
        .prompt-nav a:hover { background: #ddd; }
        .error { color: #c44; font-size: 0.85em; }
    </style>
</head>
<body>
    <h1>Sample Tracker</h1>
    <div class="run-select" id="run-select"></div>
    <div class="meta" id="status">Loading...</div>
    <div class="prompt-nav" id="nav"></div>
    <div id="content"></div>
    <div class="controls">
        <span class="status" id="poll-status"></span>
        <button onclick="refresh()">Refresh</button>
    </div>
    <script>
        let currentFile = '';

        function refresh() {
            fetch('/api/samples').then(r => r.json()).then(data => {
                const content = document.getElementById('content');
                const nav = document.getElementById('nav');
                const status = document.getElementById('status');
                const runSelect = document.getElementById('run-select');
                const steps = Object.keys(data.samples).map(Number).sort((a,b) => a-b);

                // Run selector
                if (data.available_files && data.available_files.length > 0) {
                    let opts = data.available_files.map(f => {
                        let label = f.replace('samples_', '').replace('.json', '');
                        let sel = f === data.current_file ? ' selected' : '';
                        return `<option value="${f}"${sel}>${label}</option>`;
                    }).join('');
                    runSelect.innerHTML = `<label>Run:</label><select onchange="switchRun(this.value)">${opts}</select>`;
                    currentFile = data.current_file;
                }

                let statusText = `${data.prompts.length} prompts · ${steps.length} steps · last fetch: ${data.last_fetch}`;
                if (data.current_file) statusText += ` · ${data.current_file}`;
                if (data.error) statusText += ` · <span class="error">${data.error}</span>`;
                status.innerHTML = statusText;

                nav.innerHTML = data.prompts.map((p, i) =>
                    `<a href="#prompt-${i}">${i}: ${p.substring(0, 40)}…</a>`
                ).join('');

                let html = '';
                for (let pi = 0; pi < data.prompts.length; pi++) {
                    html += `<div class="prompt-section" id="prompt-${pi}">`;
                    html += `<div class="prompt-text">${escHtml(data.prompts[pi])}</div>`;
                    html += `<div class="steps-row">`;
                    for (const step of steps) {
                        const sample = data.samples[step] && data.samples[step][pi];
                        html += `<div class="step-card">`;
                        html += `<div class="step-label">Step ${step}</div>`;
                        if (sample) {
                            html += `<div class="completion">${escHtml(sample.completion)}</div>`;
                        } else {
                            html += `<div class="no-data">No sample</div>`;
                        }
                        html += `</div>`;
                    }
                    html += `</div></div>`;
                }
                content.innerHTML = html;
            });
        }
        function switchRun(file) {
            fetch('/api/switch?file=' + encodeURIComponent(file), {method: 'POST'})
                .then(() => refresh());
        }
        function escHtml(s) {
            const d = document.createElement('div');
            d.textContent = s;
            return d.innerHTML;
        }
        refresh();
        setInterval(refresh, 10000);
    </script>
</body>
</html>
"""


def _exec(command, timeout=10):
    """Run a command on the cluster via proxy exec."""
    payload = json.dumps({"command": command, "timeout": timeout}).encode()
    req = urllib.request.Request(
        f"{PROXY_URL}/proxy/exec",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def scan_files():
    """Find all samples_*.json files in the remote dir."""
    global AVAILABLE_FILES
    try:
        data = _exec(f"find {REMOTE_DIR} -maxdepth 1 -name 'samples_*.json' -o -name 'sample_tracker.json' | xargs ls -t 2>/dev/null")
        if data["exit_code"] == 0 and data["stdout"].strip():
            paths = data["stdout"].strip().split("\n")
            AVAILABLE_FILES = [p.split("/")[-1] for p in paths]
    except Exception:
        pass


def fetch_loop():
    global SAMPLES_DATA, LAST_FETCH, FETCH_ERROR, CURRENT_FILE
    while True:
        try:
            # Periodically scan for new files
            scan_files()

            # Auto-select latest file if none selected
            if not CURRENT_FILE and AVAILABLE_FILES:
                CURRENT_FILE = AVAILABLE_FILES[0]  # ls -t sorts newest first

            if not CURRENT_FILE:
                FETCH_ERROR = "no sample files found yet"
                time.sleep(POLL_INTERVAL)
                continue

            path = f"{REMOTE_DIR}/{CURRENT_FILE}"
            data = _exec(f"cat {path}")
            if data["exit_code"] == 0 and data["stdout"].strip():
                SAMPLES_DATA = json.loads(data["stdout"])
                SAMPLES_DATA["samples"] = {
                    int(k): v for k, v in SAMPLES_DATA.get("samples", {}).items()
                }
                LAST_FETCH = time.strftime("%H:%M:%S")
                FETCH_ERROR = ""
            else:
                FETCH_ERROR = "file not found yet"
        except Exception as e:
            FETCH_ERROR = str(e)[:100]
        time.sleep(POLL_INTERVAL)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/samples")
def api_samples():
    return jsonify({
        "prompts": SAMPLES_DATA.get("prompts", []),
        "samples": SAMPLES_DATA.get("samples", {}),
        "last_fetch": LAST_FETCH,
        "error": FETCH_ERROR,
        "current_file": CURRENT_FILE,
        "available_files": AVAILABLE_FILES,
    })


@app.route("/api/switch", methods=["POST"])
def api_switch():
    global CURRENT_FILE, SAMPLES_DATA
    f = flask_request.args.get("file", "")
    if f in AVAILABLE_FILES:
        CURRENT_FILE = f
        SAMPLES_DATA = {"prompts": [], "samples": {}}
        return jsonify({"status": "switched", "file": f})
    return jsonify({"status": "not found"}), 404


def main():
    global PROXY_URL, REMOTE_DIR, POLL_INTERVAL

    parser = argparse.ArgumentParser(description="Local sample viewer")
    parser.add_argument("--proxy", default="http://10.0.0.19:2222")
    parser.add_argument("--remote-dir", default="/home/athuser/luxi-files/molloy")
    parser.add_argument("--poll-interval", type=int, default=15)
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    PROXY_URL = args.proxy
    REMOTE_DIR = args.remote_dir.rstrip("/")
    POLL_INTERVAL = args.poll_interval

    fetch_thread = threading.Thread(target=fetch_loop, daemon=True)
    fetch_thread.start()

    print(f"Sample Viewer at http://localhost:{args.port}")
    print(f"Watching {REMOTE_DIR}/samples_*.json via {PROXY_URL}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
