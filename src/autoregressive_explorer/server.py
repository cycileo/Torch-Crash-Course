import os
import sys
import threading
import time
import torch
import logging
import urllib.request
import socket
import subprocess
from flask import Flask, request, jsonify
from IPython.display import IFrame, display, HTML
from werkzeug.serving import make_server

os.environ["WERKZEUG_RUN_MAIN"] = "true"

def start_explorer(model, encode=None, decode=None, stoi=None, itos=None, port=54321, context_length=256):
    print('fix safari')
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if is_port_in_use(port):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stop", timeout=1)
        except:
            pass
        time.sleep(1)

    app = Flask(__name__)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    @app.after_request
    def add_cors(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Bypass-Tunnel-Reminder'
        return response

    @app.route("/")
    def index(): 
        return html_content

    @app.route("/_stop")
    def stop():
        if app.server_ref: 
            threading.Thread(target=app.server_ref.shutdown).start()
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
    time.sleep(1) 

    if 'google.colab' in sys.modules:
        print("Starting LocalTunnel to bypass Safari browser security...")
        
        lt_process = subprocess.Popen(
            ['npx', 'localtunnel', '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        lt_url = ""
        while True:
            line = lt_process.stdout.readline()
            if "your url is:" in line:
                lt_url = line.split("your url is:")[1].strip()
                break
                
        display(HTML(f"""
            <div style="border: 2px solid #4c1d95; border-radius: 12px; padding: 25px; background: #fdf2ff; text-align: center; font-family: sans-serif;">
                <h2 style="color: #4c1d95; margin-bottom: 10px;">🚀 Transformer Explorer Ready</h2>
                <p style="color: #6b21a8; margin-bottom: 20px;">This public link bypasses all browser security restrictions.</p>
                <div style="margin-bottom: 15px; padding: 10px; background: #fff; border-radius: 6px; font-size: 14px; color: #333;">
                    <b>Important:</b> When the new tab opens, click the blue <b>"Click to Continue"</b> button.
                </div>
                <a href="{lt_url}" target="_blank" style="background: #7c3aed; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: bold; font-size: 16px; transition: 0.3s;">
                    Open Explorer (Universal)
                </a>
            </div>
        """))
    else:
        display(IFrame(src=f"http://localhost:{port}/", width="100%", height="600px"))

    return app.server_ref