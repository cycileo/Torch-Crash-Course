import os
import torch


def load_minigpt(assets_dir, device):
    """
    Loads the default pre-trained MiniGPT model from the assets directory.

    Returns:
        (model, encode, decode, bos_token): A tuple with the loaded model,
        character-level encode/decode functions, and the BOS token id (0).
    """
    import json
    from .model import MiniGPT

    model_path = os.path.join(assets_dir, 'minigpt_weights.pth')
    chars_path = os.path.join(assets_dir, 'chars.json')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing {model_path}. Save model weights in 'assets/' first.")
    if not os.path.exists(chars_path):
        raise FileNotFoundError(f"Missing {chars_path}. Save chars.json in 'assets/' first.")

    with open(chars_path, 'r', encoding='utf-8') as f:
        chars = json.load(f)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    # MiniGPT was trained on lowercase text — lowercase inside encode
    encode = lambda s: [stoi.get(c, 0) for c in s.lower()]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])

    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = MiniGPT(vocab_size=len(chars), block_size=64)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, encode, decode, 0  # BOS token = 0 (padding/start token)


def load_hf_model(model_name, device):
    """
    Loads any HuggingFace causal language model by name.
    The model is downloaded to the HuggingFace cache (not stored in assets/).

    Args:
        model_name (str): A HuggingFace model identifier, e.g. 'Qwen/Qwen3-0.6B'.
        device: A torch device to load the model onto.

    Returns:
        (model, encode, decode, bos_token): A tuple with the loaded model,
        tokenizer-based encode/decode functions, and the tokenizer's BOS token id.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, logging as hf_logging
    except ImportError:
        raise ImportError("The 'transformers' package is required for HuggingFace backends. "
                          "Install it with: pip install transformers")

    # Suppress noisy HF warnings
    import logging
    hf_logging.set_verbosity_error()
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    print(f"Loading '{model_name}' from HuggingFace (this may take a moment the first time)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    model.to(device)
    model.eval()

    def encode(s):
        return tokenizer.encode(s, add_special_tokens=False)

    def decode(token_ids):
        # convert_tokens_to_string avoids byte-fallback '?' for individual tokens
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        text = tokenizer.convert_tokens_to_string(tokens)
        # Replace GPT-2's Ġ (space prefix) with a real space and strip leading space
        return text

    # Use the tokenizer's BOS id, fallback to eos or 0
    bos_token = tokenizer.bos_token_id or tokenizer.eos_token_id or 0

    return model, encode, decode, bos_token


# Shorthand aliases for well-known models
HF_ALIASES = {
    'qwen': 'Qwen/Qwen3-0.6B',
    'gpt2': 'gpt2',
}
