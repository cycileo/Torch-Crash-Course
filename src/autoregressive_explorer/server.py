import os
import sys
import threading
import time
import torch
import logging
import urllib.request
import socket
from flask import Flask, request, jsonify
from IPython.display import IFrame, display, HTML
from werkzeug.serving import make_server

# Suppress Flask logs
os.environ["WERKZEUG_RUN_MAIN"] = "true"

def start_explorer(model, encode=None, decode=None, stoi=None, itos=None, port=54321, context_length=256):
    # 1. Kill old server
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if is_port_in_use(port):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stop", timeout=1)
        except:
            pass
        time.sleep(1)

    # 2. Flask Setup
    app = Flask(__name__)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    @app.after_request
    def add_cors(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.route("/")
    def index(): return html_content

    @app.route("/_stop")
    def stop():
        if app.server_ref: threading.Thread(target=app.server_ref.shutdown).start()
        return "Shutting down"

    @app.route("/get_logits", methods=["POST"])
    def get_logits():
        data = request.json
        text = data.get("text", "").lower()
        try:
            tokens = encode(text) if encode else [stoi.get(c, 0) for c in text]
            max_len = getattr(model, 'block_size', context_length)
            tokens = tokens[-max_len:]
            device = next(model.parameters()).device
            model.eval()
            with torch.no_grad():
                x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
            last_logits = logits[0, -1, :]
            topk_vals, topk_idx = torch.topk(last_logits, 10)
            chars = [decode([i.item()]) if decode else itos.get(i.item(), '?') for i in topk_idx]
            return jsonify({"top10": [{"char": c, "logit": v.item()} for c, v in zip(chars, topk_vals)]})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def serve():
        app.server_ref = make_server('0.0.0.0', port, app)
        app.server_ref.serve_forever()

    app.server_ref = None
    threading.Thread(target=serve, daemon=True).start()
    time.sleep(2) # Give it extra time to boot

    # 3. THE COLAB FIX: Dynamic URL + Minimal Link
    if 'google.colab' in sys.modules:
        from google.colab.output import eval_js
        from google.colab import output
        try:
            proxy_url = eval_js(f"google.colab.kernel.proxyPort({port})")
            # A sleek, minimal link right above the iframe
            display(HTML(f"""
                <div style="text-align: right; margin-bottom: 8px; font-family: sans-serif;">
                    <a href="{proxy_url}" target="_blank" style="background: #f8fafc; color: #475569; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; border: 1px solid #cbd5e1; transition: 0.2s;">
                        ↗️ Open in New Tab (Fullscreen)
                    </a>
                </div>
            """))
        except Exception as e:
            pass
        
        output.serve_kernel_port_as_iframe(port, height='600')
    else:
        display(IFrame(src=f"http://localhost:{port}/", width="100%", height="600px"))

    return app.server_ref