import math
import os
import torch
import torch.nn as nn
import gradio as gr
import tiktoken
from dataclasses import dataclass
from torch.nn import functional as F


# -----------------------------
# Model definition (must match training checkpoint)
# -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int
    vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# Load model once (cached)
# -----------------------------

CHECKPOINT_PATH = os.environ.get("MODEL_PATH", "trained_model.pt")

def pick_device():
    # Spaces usually runs on CPU unless you enable GPU Space.
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = pick_device()
ENC = tiktoken.get_encoding("gpt2")

_MODEL = None

def load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    # Important: your checkpoint includes a pickled GPTConfig object
    # so we must load with weights_only=False.
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    config = ckpt["config"]  # expected to be GPTConfig object
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    _MODEL = model
    return _MODEL


@torch.no_grad()
def generate_text(prompt: str, max_new_tokens: int, top_k: int, temperature: float):
    model = load_model()

    prompt = (prompt or "").strip()
    if not prompt:
        return "Please enter a prompt."

    idx = torch.tensor(ENC.encode(prompt), dtype=torch.long, device=DEVICE)[None, :]

    for _ in range(max_new_tokens):
        logits = model(idx)[:, -1, :]  # last position
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        k = min(top_k, probs.size(-1))
        topk_probs, topk_idx = torch.topk(probs, k=k, dim=-1)
        next_ix = torch.multinomial(topk_probs, 1)
        next_token = torch.gather(topk_idx, -1, next_ix)

        idx = torch.cat([idx, next_token], dim=1)

    return ENC.decode(idx[0].tolist())


# -----------------------------
# Gradio UI
# -----------------------------

with gr.Blocks(title="Decoder-only GPT-2 (124M) - Fine-tuned") as demo:
    gr.Markdown(
        """
# Decoder-only GPT-2 (124M) — Fine-tuned on Custom Corpus

Enter a prompt and generate text from the fine-tuned model checkpoint.
"""
    )

    with gr.Row():
        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Type your prompt here...",
            lines=4,
        )

    with gr.Row():
        max_new_tokens = gr.Slider(10, 300, value=120, step=1, label="Max new tokens")
        top_k = gr.Slider(1, 200, value=50, step=1, label="Top-k")
        temperature = gr.Slider(0.2, 2.0, value=1.0, step=0.05, label="Temperature")

    btn = gr.Button("Generate")
    output = gr.Textbox(label="Output", lines=12)

    btn.click(
        fn=generate_text,
        inputs=[prompt, max_new_tokens, top_k, temperature],
        outputs=[output],
    )

    gr.Markdown(f"**Device:** `{DEVICE}`")

demo.launch()
