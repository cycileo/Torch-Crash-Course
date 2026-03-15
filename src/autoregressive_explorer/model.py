import torch
import torch.nn as nn
import lightning as L

class MiniGPT(L.LightningModule):
    def __init__(self, vocab_size, block_size, n_embd=64):
        super().__init__()
        self.block_size = block_size
        
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=n_embd, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx) 
        pos_emb = self.position_embedding(torch.arange(T, device=self.device)) 
        x = tok_emb + pos_emb
        
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        
        logits = self.lm_head(x) 
        return logits
