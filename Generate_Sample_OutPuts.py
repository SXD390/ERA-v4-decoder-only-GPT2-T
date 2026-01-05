import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass
import tiktoken


# -----------------------------
# Model definition (must match training)
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
        assert T <= self.config.block_size

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# Text generation
# -----------------------------

@torch.no_grad()
def generate(
    model,
    device,
    enc,
    prompt,
    max_new_tokens=120,
    top_k=50,
):
    model.eval()

    tokens = enc.encode(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]

    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :]  # last token
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_idx = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        next_token = torch.gather(topk_idx, -1, ix)

        idx = torch.cat([idx, next_token], dim=1)

    return enc.decode(idx[0].tolist())


# -----------------------------
# Main
# -----------------------------

def main():
    # Device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load("trained_model.pt", map_location=device, weights_only=False)
    config = ckpt["config"]
    model = GPT(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    enc = tiktoken.get_encoding("gpt2")

    print("\nModel loaded successfully.")
    print("Type a prompt and press Enter (Ctrl+C to exit)\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()
            if not prompt:
                continue

            output = generate(
                model=model,
                device=device,
                enc=enc,
                prompt=prompt,
            )

            print("\n--- Generated ---")
            print(output)
            print("-----------------\n")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
