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

import gc

os.environ["WERKZEUG_RUN_MAIN"] = "true"

# Global state for resource management
_MODEL_CACHE = {}    # { cache_key: (model, encode, decode, bos_token, default_seed, label_width) }
_SERVER_REGISTRY = {}  # { cache_key: server }  -- tracks which server uses which model

def start_explorer(model=None, encode=None, decode=None, stoi=None, itos=None,
                   backend='minigpt', port=None, context_length=256, evict_others=False, use_hf_cache=False):
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
    evict_others (bool, optional): If True, evicts any other cached HF models from memory 
        and shuts down their servers before launching. Defaults to False.
    use_hf_cache (bool, optional): If True, enables HuggingFace's internal KV cache for 
        faster inference at the cost of higher memory usage. Defaults to False.

    Returns:
    server: A reference to the running Werkzeug server instance.
    """
    global _MODEL_CACHE, _SERVER_REGISTRY

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assets_dir = os.path.join(base_dir, 'assets')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Create a unique key for caching
    if model is None:
        from .backends import HF_ALIASES
        cache_key = HF_ALIASES.get(backend, backend)
        is_hf = (cache_key != 'minigpt')
    else:
        cache_key = id(model)
        is_hf = False

    # 2. Check cache
    if cache_key in _MODEL_CACHE:
        model, encode, decode, bos_token, default_seed, label_width = _MODEL_CACHE[cache_key]
        model.to(device)
    else:
        # Evict other cached HF models if requested OR when loading a completely new HF model
        if is_hf and evict_others:
            hf_keys_to_evict = [k for k in _MODEL_CACHE if k not in ('minigpt',) and not isinstance(k, int) and k != cache_key]
            for k in hf_keys_to_evict:
                # Kill the server that was using this model before freeing the model
                if k in _SERVER_REGISTRY:
                    try:
                        _SERVER_REGISTRY[k].shutdown()
                    except Exception:
                        pass
                    del _SERVER_REGISTRY[k]
                evicted_model = _MODEL_CACHE.pop(k)[0]
                evicted_model.cpu()
                del evicted_model
                gc.collect()
                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                elif torch.cuda.is_available(): torch.cuda.empty_cache()

        # Load model logic (existing logic)
        if model is None:
            from .backends import load_minigpt, load_hf_model, HF_ALIASES
            resolved = HF_ALIASES.get(backend, backend)
            if resolved == 'minigpt':
                model, encode, decode, bos_token = load_minigpt(assets_dir, device)
                default_seed = 'o romeo'
                label_width = '35px'
            else:
                model, encode, decode, bos_token = load_hf_model(resolved, device)
                default_seed = 'Once upon a time'
                label_width = '85px'
        else:
            # User passed their own model — ensure it is on the right device
            model.to(device)
            model.eval()
            bos_token = 0  # sensible default for custom models
            default_seed = 'o romeo'  # custom models assumed to be MiniGPT-like
            label_width = '35px'
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
        
        # Save to cache
        _MODEL_CACHE[cache_key] = (model, encode, decode, bos_token, default_seed, label_width)

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
                port = None  # Force a new port if we couldn't clear the requested one

    if port is None:
        port = random.randint(10240, 65535)
        while is_port_in_use(port):
            port = random.randint(10240, 65535)

    app = Flask(__name__)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read().replace('{{DEFAULT_SEED}}', default_seed).replace('{{LABEL_WIDTH}}', label_width)

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
        temp = data.get("temperature", 1.0)
        top_k = data.get("top_k", 10)
        sample_from_top_k = data.get("sample_from_top_k", False)
        return_all = data.get("return_all", False)
        try:
            tokens = encode(text)
            # Always prepend the BOS token to get unconditional probability for the first char
            tokens = [bos_token] + tokens
            max_len = getattr(model, 'block_size', getattr(getattr(model, 'config', None), 'max_position_embeddings', context_length))
            tokens = tokens[-max_len:]
            
            with torch.no_grad():
                x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                # use_hf_cache controls HF's internal KV cache (off by default to reduce memory)
                call_kwargs = {'use_cache': use_hf_cache} if hasattr(model, 'config') else {}
                raw = model(x, **call_kwargs)
                # Unpack: raw tensor, tuple, or HuggingFace model output object
                if isinstance(raw, tuple):
                    logits = raw[0]
                elif hasattr(raw, 'logits'):
                    logits = raw.logits
                else:
                    logits = raw
                del raw  # Free the output wrapper / extra tensors immediately
                del x

            def process_logits(logits_1d, t):
                # Apply top-k filtering for sampling ONLY if requested
                if sample_from_top_k:
                    v, _ = torch.topk(logits_1d, min(top_k, logits_1d.size(-1)))
                    logits_1d[logits_1d < v[..., [-1]]] = -float('Inf')

                # Handle greedy sampling separately
                if t <= 1e-6:
                    probs = torch.softmax(logits_1d * 1e6, dim=-1) # Focus on max
                else:
                    probs = torch.softmax(logits_1d / t, dim=-1)
                
                # Sample from (possibly truncated) distribution
                sampled_idx = torch.multinomial(probs, 1).item()
                sampled_char = decode([sampled_idx])
                
                # Extract Top K for visualization
                # Note: we use top_k to control display, regardless of sampling strategy
                tk_probs, tk_idx = torch.topk(probs, min(top_k, probs.size(-1)))
                top10 = [{"char": decode([i.item()]), "prob": p.item()} for p, i in zip(tk_probs, tk_idx)]
                return top10, sampled_char

            if return_all:
                res_list = [process_logits(logits[0, t, :].clone(), temp) for t in range(logits.size(1))]
                payloads = [r[0] for r in res_list]
                token_strings = [decode([t]) for t in tokens[1:]]
                del logits  # Free the large [1, seq_len, vocab_size] tensor now
                return jsonify({
                    "top10_all": payloads,
                    "tokens": token_strings
                })

            last_logits = logits[0, -1, :].clone()
            del logits  # Free the large tensor before building the response
            top10, sampled_char = process_logits(last_logits, temp)
            return jsonify({
                "top10": top10, 
                "sampled_char": sampled_char
            })
        except Exception as e:
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    @app.route("/reset_memory", methods=["POST", "OPTIONS"])
    def reset_memory():
        if request.method == "OPTIONS":
            return jsonify({}), 200
        gc.collect()
        if torch.backends.mps.is_available(): torch.mps.empty_cache()
        elif torch.cuda.is_available(): torch.cuda.empty_cache()
        return jsonify({"status": "ok"})

    def serve():
        global _SERVER_REGISTRY
        srv = make_server('0.0.0.0', port, app)
        _SERVER_REGISTRY[cache_key] = srv
        app.server_ref = srv
        srv.serve_forever()

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