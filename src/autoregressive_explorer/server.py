import os
import sys
import threading
import time
import torch
import logging
import urllib.request
import socket
import random
from flask import Flask, request, jsonify
from IPython.display import IFrame, display, HTML
from werkzeug.serving import make_server

os.environ["WERKZEUG_RUN_MAIN"] = "true"

def start_explorer(model=None, encode=None, decode=None, stoi=None, itos=None, port=None, context_length=256):
    """
    Launches an interactive web-based visualization tool for exploring the autoregressive 
    predictions of a language model.
    
    This function starts a background Flask server that serves a UI for analyzing 
    token probabilities, top-k predictions, and conditional distributions.
    
    Parameters:
    model (nn.Module, optional): A trained PyTorch model. If None, the function attempts 
        to load a pre-trained model and weights from the 'assets/' directory. 
        Pass your own model instance to test local training results.
    encode (callable, optional): A text-to-integers encoder function.
    decode (callable, optional): An integers-to-text decoder function.
    stoi (dict, optional): A string-to-index mapping dictionary.
    itos (dict, optional): An index-to-string mapping dictionary.
        Note: You can provide custom vocabulary handlers (encode/decode or stoi/itos) 
        to experiment with different tokenization strategies. If omitted, they are 
        automatically generated from 'assets/chars.json'.
    port (int, optional): The network port for the explorer server. If None, a random 
        free port is chosen.
    context_length (int, optional): The maximum number of tokens to feed into the model. 
        Defaults to 256 or the model's 'block_size' attribute if found.

    Returns:
    server: A reference to the running Werkzeug server instance.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_dir = os.path.join(base_dir, 'assets')
    chars_path = os.path.join(assets_dir, 'chars.json')
    model_path = os.path.join(assets_dir, 'minigpt_weights.pth')

    # Load chars if encode/decode or stoi/itos are missing, OR if model needs to be loaded (to get vocab_size)
    if (encode is None and stoi is None) or (decode is None and itos is None) or (model is None):
        import json
        if not os.path.exists(chars_path):
            raise FileNotFoundError(f"Missing {chars_path}. Please provide encode/decode or stoi/itos and/or save chars in 'assets/' first.")
        with open(chars_path, 'r', encoding='utf-8') as f:
            chars = json.load(f)

        if encode is None and stoi is None: 
            stoi = { ch:i for i,ch in enumerate(chars) }
        if decode is None and itos is None: 
            itos = { i:ch for i,ch in enumerate(chars) }

    # Generate default encode/decode functions if missing to keep the main loop fast
    if encode is None: 
        encode = lambda s: [stoi.get(c, 0) for c in s]
    if decode is None: 
        decode = lambda l: ''.join([itos.get(i, '?') for i in l])

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Auto-load logic if model isn't provided
    if model is None:
        from .model import MiniGPT
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing {model_path}. Please provide a model, or save weights in 'assets/' first.")
            
        loaded_data = torch.load(model_path, map_location=device, weights_only=True)
        
        model = MiniGPT(vocab_size=len(chars), block_size=64)
        model.load_state_dict(loaded_data)
        
    model.to(device)
    model.eval()

    # print('debug')
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if port is not None:
        if is_port_in_use(port):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/_stop", timeout=1)
            except Exception:
                pass
            for _ in range(10):
                if not is_port_in_use(port):
                    break
                time.sleep(0.5)
            if is_port_in_use(port):
                port = None

    if port is None:
        port = random.randint(10240, 65535)
        while is_port_in_use(port):
            port = random.randint(10240, 65535)

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
    def index(): 
        return html_content

    @app.route("/_stop")
    def stop():
        if app.server_ref: 
            threading.Thread(target=app.server_ref.shutdown).start()
        return "Shutting down"

    @app.route("/get_logits", methods=["POST", "OPTIONS"])
    def get_logits():
        if request.method == "OPTIONS":
            return jsonify({}), 200

        data = request.json
        text = data.get("text", "").lower()
        return_all = data.get("return_all", False)
        try:
            tokens = encode(text)
            # Always prepend [0] (pad/BOS token) to get the unconditional probability for the first char
            tokens = [0] + tokens
            max_len = getattr(model, 'block_size', context_length)
            tokens = tokens[-max_len:]
            
            with torch.no_grad():
                x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                if isinstance(logits, tuple): logits = logits[0]
                
            def get_top10(logits_1d):
                topk_vals, topk_idx = torch.topk(logits_1d, 10)
                chars = [decode([i.item()]) for i in topk_idx]
                return [{"char": c, "logit": v.item()} for c, v in zip(chars, topk_vals)]

            if return_all:
                all_payloads = [get_top10(logits[0, t, :]) for t in range(logits.size(1))]
                return jsonify({"top10_all": all_payloads})

            last_logits = logits[0, -1, :]
            return jsonify({"top10": get_top10(last_logits)})
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
            display(HTML(f"""
                <div style="text-align: right; margin-bottom: 8px; font-family: sans-serif;">
                    <a href="{proxy_url}" target="_blank" style="background: #f8fafc; color: #475569; padding: 6px 12px; border-radius: 6px; text-decoration: none; font-size: 12px; border: 1px solid #cbd5e1; transition: 0.2s;">
                        ↗️ Open in New Tab (Fullscreen)
                    </a>
                </div>
            """))
        except Exception:
            pass
        
        output.serve_kernel_port_as_iframe(port, height='600')
    else:
        display(IFrame(src=f"http://localhost:{port}/", width="100%", height="600px"))

    return app.server_ref