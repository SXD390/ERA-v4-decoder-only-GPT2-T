
# Decoder-only GPT-2 Style Transformer (124M)

This repository implements, trains, and deploys a **decoder-only Transformer (GPT-2 small, ~124M parameters)** on a custom text corpus.  
The model is trained until the **cross-entropy loss drops below 0.099999**, and is deployed as an interactive **Hugging Face Spaces app** for text generation.

---

## Architecture

The model follows the standard **GPT-2 decoder-only Transformer** architecture with causal self-attention and autoregressive next-token prediction.

```mermaid
flowchart TB
    A[Input Tokens] --> B[Token Embedding]
    P[Position Index] --> C[Positional Embedding]
    B --> D[Add]
    C --> D

    D --> E[Transformer Block × 12]

    subgraph BLOCK[Transformer Block]
        LN1[LayerNorm] --> SA[Causal Self-Attention]
        SA --> R1[Residual Add]
        R1 --> LN2[LayerNorm]
        LN2 --> FFN[MLP: Linear → GELU → Linear]
        FFN --> R2[Residual Add]
    end

    E --> LNf[Final LayerNorm]
    LNf --> LM[LM Head with tied weights]
    LM --> O[Logits]
    O --> CE[Cross-Entropy Loss]
````

**Key characteristics**

* Decoder-only (no encoder)
* Causal masking (no access to future tokens)
* Weight tying between token embeddings and output projection
* GPT-2 small configuration (12 layers, 12 heads, 768 hidden dim)

---

## Model Details

| Component      | Value                     |
| -------------- | ------------------------- |
| Parameters     | **124,439,808**           |
| Vocabulary     | GPT-2 BPE (50,257 tokens) |
| Context Length | 1024                      |
| Layers         | 12                        |
| Heads          | 12                        |
| Hidden Size    | 768                       |

---

## Training Setup

* **Dataset**: `input.txt`[Link](https://raw.githubusercontent.com/SXD390/ERA-v4-decoder-only-GPT2-T/refs/heads/main/input.txt) (~338k tokens)
* **Objective**: Next-token prediction (autoregressive LM)
* **Optimizer**: AdamW
* **Scheduler**: OneCycleLR
* **Gradient Clipping**: 0.5
* **Batch / Sequence**: B=16, T=128
* **Device**: Apple M-series (MPS) / CPU / CUDA

Training was continued until the loss crossed the threshold 0.09.

---

## Training Results

* **Target loss**: `< 0.099999`
* **Best achieved loss**: **0.0860** 

### Training log excerpt

```
step 5900 | loss: 0.1410 | avg_loss: 0.1349 | best_loss: 0.0890
step 6100 | loss: 0.1301 | avg_loss: 0.1242 | best_loss: 0.0860
```

Full logs are available here - [Training.log](https://raw.githubusercontent.com/SXD390/ERA-v4-decoder-only-GPT2-T/refs/heads/main/Terminal_Logs.log)

---

## Sample Outputs

# SAMPLE 1

```bash
% python3 Generate_Sample_OutPuts.py
Using device: mps

Model loaded successfully.
Type a prompt and press Enter (Ctrl+C to exit)

Prompt> We are accounted poor citizens, the
```

# Output 1
```log
--- Generated ---
We are accounted poor citizens, the haply find a throne,
And yest once Clarence, being seated, so soon-door thoughts back,
she cannot speak thine eyes in my heart-wingided them no loss.

RICHARD:
Stay, here, my lord! the king,

QUEEN ELIZABETH:
 didst thou thought it as she have his part;
And soon off thy brother, but open.

KING HENRY VI:
Say, but not, being but not before him to prison,
LUCENTIO:
Clare
-----------------
```

# SAMPLE 2

```bash
Prompt> Elizabeth
```

# Output 2
```log
--- Generated ---
Elizabeth
 we should be the common people,
And in this morningers would quaff till we know.

First Keeper:
The common eyes! here do't perceive

MAMILLO:
If he were ay, here?

Second Gentleman:
Yet he were so heProdLEY:
 friends, I hate, in heaven;
The clamour of all in the harmony of it,
Whilst I would not would have given me.

First Citizen:
Ay, and therefore, to see the churchits to have an hour;
after
-----------------

Prompt> 
```

---

## Hugging Face Spaces Demo

The trained model is deployed as a **Hugging Face Spaces (Gradio) app**, allowing interactive prompt-based generation.

### App : https://huggingface.co/spaces/SXD390/GPT2_NON_SFT

📸 **Screenshot**
![](https://github.com/SXD390/ERA-v4-decoder-only-GPT2-T/blob/main/hf_space/HF_app_SC.png)


---

## Repository Structure

```
.
├── Decoder_only_GPT2_style_transformer.py   # Training script
├── Generate_Sample_OutPuts.py               # Standalone inference script
├── input.txt                                # Training corpus
├── trained_model.pt                         # Trained model checkpoint
├── Terminal_Logs.log                        # Training logs
├── outputs.md                               # Sample generations
└── hf_space/                                # Hugging Face Spaces app
    ├── app.py
    ├── requirements.txt
    └── README.md
```

---

## How to Run Locally

### Install dependencies

```bash
pip install torch transformers tiktoken tqdm gradio
```

### Generate text from the trained model

```bash
python3 Generate_Sample_OutPuts.py
```

---

## Notes

* The extremely low loss threshold implies **strong fitting to the provided corpus**, which is expected given the assignment objective.
* The model is trained and evaluated purely as a **decoder-only language model**, without any encoder or bidirectional context.
* The Hugging Face deployment demonstrates successful model loading, inference, and interactive generation.

---

