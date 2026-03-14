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
        from google.colab.output import eval_js
        from google.colab import output
        
        try:
            proxy_url = eval_js(f"google.colab.kernel.proxyPort({port})")
        except Exception:
            proxy_url = "#"
        
        try:
            req = urllib.request.Request('https://loca.lt/mytunnelpassword')
            req.add_header('User-Agent', 'Mozilla/5.0')
            tunnel_password = urllib.request.urlopen(req).read().decode('utf8').strip()
        except Exception:
            try:
                tunnel_password = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
            except:
                tunnel_password = "Error fetching IP"
        
        lt_process = subprocess.Popen(
            ['npx', 'localtunnel', '--port', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        lt_url = "#"
        while True:
            line = lt_process.stdout.readline()
            if "your url is:" in line:
                lt_url = line.split("your url is:")[1].strip()
                break
                
        display(HTML(f"""
            <div style="display: flex; gap: 15px; margin-bottom: 10px; font-family: sans-serif; max-width: 800px;">
                <div style="flex: 1; padding: 15px; border: 1px solid #cbd5e1; border-radius: 8px; background: #f8fafc;">
                    <h4 style="margin: 0 0 8px 0; color: #0f172a; font-size: 14px;">Option 1: Standard Tab</h4>
                    <p style="margin: 0 0 12px 0; font-size: 12px; color: #475569;">Best for Chrome. Fast and secure.</p>
                    <a href="{proxy_url}" target="_blank" style="display: inline-block; background: #2563eb; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: bold;">↗️ Open Standard</a>
                </div>
                <div style="flex: 1; padding: 15px; border: 1px solid #c084fc; border-radius: 8px; background: #faf5ff;">
                    <h4 style="margin: 0 0 8px 0; color: #4c1d95; font-size: 14px;">Option 2: Universal (Safari)</h4>
                    <p style="margin: 0 0 12px 0; font-size: 12px; color: #6b21a8;">If Option 1 fails. Password: <code style="background: #e9d5ff; padding: 2px 6px; border-radius: 4px; color: #7e22ce; font-weight: bold;">{tunnel_password}</code></p>
                    <a href="{lt_url}" target="_blank" style="display: inline-block; background: #9333ea; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; font-weight: bold;">🌐 Open Universal</a>
                </div>
            </div>
        """))
        
        output.serve_kernel_port_as_iframe(port, height='600')
    else:
        display(IFrame(src=f"http://localhost:{port}/", width="100%", height="600px"))

    return app.server_ref