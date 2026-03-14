import os
import sys
import threading
import time
import torch
import logging
import urllib.request
import socket
from flask import Flask, request, jsonify
from IPython.display import IFrame, display
from werkzeug.serving import make_server

# Completely suppress Flask/Werkzeug logs and banners
os.environ["WERKZEUG_RUN_MAIN"] = "true"
cli = sys.modules.get('flask.cli', None)
if cli is not None:
    cli.show_server_banner = lambda *x: None

def start_explorer(model, encode=None, decode=None, stoi=None, itos=None, port=54321, context_length=256):
    """
    Starts a background Flask server and displays the Autoregressive Explorer UI in the notebook.
    """
    # 1. ATTEMPT TO KILL ANY EXISTING FLASK SERVER ON THIS PORT!
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if is_port_in_use(port):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/_stop", timeout=1)
        except Exception:
            pass
        
        # Wait for port to actually free up
        for _ in range(10):
            if not is_port_in_use(port):
                break
            time.sleep(0.5)

    # 2. Initialize fresh Flask app
    app = Flask(__name__)
    app.logger.disabled = True
    logging.getLogger('werkzeug').disabled = True
    
    # Read the HTML content
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'OPTIONS,POST,GET')
        return response

    @app.route("/")
    def index():
        return html_content

    # We store the werkzeug server reference here so the endpoint can access it
    app.server_ref = None

    @app.route("/_stop")
    def stop():
        if app.server_ref is not None:
            threading.Thread(target=app.server_ref.shutdown, daemon=True).start()
        return "Shutting down"

    @app.route("/get_logits", methods=["POST", "OPTIONS"])
    def get_logits():
        if request.method == "OPTIONS":
            return jsonify({}), 200

        data = request.json
        text = data.get("text", "")
        
        try:
            # 1. Tokenize string -> indices
            if encode is not None:
                encoded = encode(text)
            elif stoi is not None:
                encoded = [stoi.get(c, 0) for c in text]
            else:
                raise ValueError("No tokenizer found. Please provide an `encode` function or `stoi` dict.")

            # Prioritize the model's explicit block_size if it exists, otherwise use fallback context_length
            if hasattr(model, 'block_size'):
                max_len = model.block_size
            else:
                max_len = context_length
                
            encoded = encoded[-max_len:]

            # 2. Model Device Check
            device = next(model.parameters()).device
            
            # 3. Forward Pass 
            model.eval()
            with torch.no_grad():
                x = torch.tensor(encoded, dtype=torch.long, device=device).unsqueeze(0)  # Shape (1, T)
                preds = model(x)
                logits = preds[0] if isinstance(preds, tuple) else preds 
                
            # 4. Extract raw T=1.0 logits for ONLY the very last token
            last_token_logits = logits[0, -1, :] # Shape: (vocab_size,)
            
            # 5. Extract Top 10 indices and values
            topk_logits, topk_indices = torch.topk(last_token_logits, 10)
            
            topk_logits = topk_logits.cpu().tolist()
            topk_indices = topk_indices.cpu().tolist()
            
            # 6. Decode indices -> string characters
            chars = []
            if decode is not None:
                chars = [decode([idx]) for idx in topk_indices]
            elif itos is not None:
                chars = [itos.get(idx, '?') for idx in topk_indices]
            else:
                chars = [chr(idx) for idx in topk_indices] # ASCII fallback

            # 7. Return formatted payload
            top10_payload = [{"char": c, "logit": l} for c, l in zip(chars, topk_logits)]
            return jsonify({"top10": top10_payload})
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    def serve_app():
        srv = make_server('0.0.0.0', port, app)
        app.server_ref = srv
        srv.serve_forever()

    # Start server in daemonized thread
    server_thread = threading.Thread(target=serve_app, daemon=True)
    server_thread.start()
    
    time.sleep(1) # Let server boot

    # FIX: Display safely handling both Colab and Local
    if 'google.colab' in sys.modules:
        from google.colab import output
        output.serve_kernel_port_as_iframe(port, height="450")
    else:
        display(IFrame(src=f"http://localhost:{port}/", width="100%", height="450px"))

    return server_thread