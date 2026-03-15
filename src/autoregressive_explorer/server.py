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

def start_explorer(model=None, encode=None, decode=None, stoi=None, itos=None,
                   backend='minigpt', port=None, context_length=256):
    """
    Launches an interactive web-based visualization tool for exploring the autoregressive 
    predictions of a language model.
    
    This function starts a background Flask server that serves a UI for analyzing 
    token probabilities, top-k predictions, and conditional distributions.
    
    Parameters:
    model (nn.Module, optional): A trained PyTorch model to use directly.
        If None, one is loaded according to the 'backend' argument.
    encode (callable, optional): A text-to-integers encoder function.
    decode (callable, optional): An integers-to-text decoder function.
    stoi (dict, optional): A string-to-index mapping dictionary.
    itos (dict, optional): An index-to-string mapping dictionary.
        Note: You can provide custom vocabulary handlers (encode/decode or stoi/itos) 
        to experiment with different tokenization strategies. If omitted, they are 
        automatically generated based on the selected backend.
    backend (str, optional): Which model to load when 'model' is None. Options:
        - 'minigpt' (default): loads the pre-trained MiniGPT from assets/.
        - 'qwen': loads Qwen3-0.6B from HuggingFace.
        - Any HuggingFace model ID string, e.g. 'gpt2'.
    port (int, optional): The network port for the explorer server. If None, a random 
        free port is chosen.
    context_length (int, optional): The maximum number of tokens to feed into the model. 
        Defaults to 256 or the model's 'block_size' attribute if found.

    Returns:
    server: A reference to the running Werkzeug server instance.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_dir = os.path.join(base_dir, 'assets')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # If model is not provided, delegate loading to backends
    if model is None:
        from .backends import load_minigpt, load_hf_model, HF_ALIASES
        resolved = HF_ALIASES.get(backend, backend)
        if resolved == 'minigpt':
            model, encode, decode, bos_token = load_minigpt(assets_dir, device)
            default_seed = 'o romeo'
        else:
            model, encode, decode, bos_token = load_hf_model(resolved, device)
            default_seed = 'Once upon a time'
    else:
        # User passed their own model — ensure it is on the right device
        model.to(device)
        model.eval()
        bos_token = 0  # sensible default for custom models
        default_seed = 'o romeo'  # custom models assumed to be MiniGPT-like
        # If no vocabulary handlers provided, fall back to the MiniGPT chars from assets/
        if encode is None and stoi is None and decode is None and itos is None:
            from .backends import load_minigpt
            _, encode, decode, bos_token = load_minigpt(assets_dir, device)
        else:
            # Build encode/decode from stoi/itos if only one side is missing
            if encode is None:
                if stoi is None:
                    raise ValueError("Please provide 'encode' or 'stoi' when passing a custom model.")
                encode = lambda s: [stoi.get(c, 0) for c in s]
            if decode is None:
                if itos is None:
                    raise ValueError("Please provide 'decode' or 'itos' when passing a custom model.")
                decode = lambda l: ''.join([itos.get(i, '?') for i in l])

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
        html_content = f.read().replace('{{DEFAULT_SEED}}', default_seed)

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
        text = data.get("text", "")
        return_all = data.get("return_all", False)
        try:
            tokens = encode(text)
            # Always prepend the BOS token to get unconditional probability for the first char
            tokens = [bos_token] + tokens
            max_len = getattr(model, 'block_size', getattr(getattr(model, 'config', None), 'max_position_embeddings', context_length))
            tokens = tokens[-max_len:]
            
            with torch.no_grad():
                x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                logits = model(x)
                # Unpack: raw tensor, tuple, or HuggingFace model output object
                if isinstance(logits, tuple):
                    logits = logits[0]
                elif hasattr(logits, 'logits'):
                    logits = logits.logits
                
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
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

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