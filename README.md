# ANLP Assignment 1 — Transformers from Scratch, Architectural Variants, and BLT

Custom encoder–decoder Transformers built from fundamental PyTorch operations, trained to
decrypt a repeating-key XOR cipher (`cipher[i] = plaintext[i] XOR key[i mod 8]`, key `ANLP2026`)
back into plaintext. Five configurations are compared in a controlled single-component ablation.

No `nn.Transformer` or `nn.MultiheadAttention` is used anywhere; attention, positional
encodings, normalization, and the BLT local encoder/decoder are all implemented from scratch.

---

## Configurations

| Config | Change from base | Positional Encoding | Attention | Normalization | Tokenization |
|---|---|---|---|---|---|
| C1 | — (base) | Sinusoidal | MHA | LayerNorm | Subword (BPE) |
| C2 | Positional encoding | RoPE | MHA | LayerNorm | Subword (BPE) |
| C3 | Attention | Sinusoidal | GQA | LayerNorm | Subword (BPE) |
| C4 | Normalization | Sinusoidal | MHA | RMSNorm | Subword (BPE) |
| C5 | Tokenization | Sinusoidal | MHA | LayerNorm | BLT (token-free) |

---

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (experiments were run on an NVIDIA RTX 5060 Ti)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is not present, install directly:

```bash
pip install torch tqdm wandb matplotlib python-Levenshtein nltk rouge-score huggingface_hub
```

### Weights & Biases

Training logs to W&B under the project `anlp-assignment1`. Log in once before training:

```bash
wandb login
```

To run without logging, set `WANDB_MODE=disabled` in the environment.

### Dataset

Place the provided dataset in `Dataset_A1/` at the project root:

```
Dataset_A1/
├── brown_cipher.txt      # binary ciphertext, one sequence per line
└── brown_plain.txt       # corresponding plaintext, one sequence per line
split_indices.json        # fixed 70/15/15 train/val/test indices
```

Paths are set at the top of `src/train.py` via the `ROOT` constant — update it to match
your local directory before running.

---

## Running

### Training

Set `CONFIG` at the top of `src/train.py` to one of `C1`, `C2`, `C3`, `C4`, `C5`, then:

```bash
python src/train.py
```

Each run writes to `outputs/<config>/`:

| File | Contents |
|---|---|
| `best.pt` | best checkpoint by validation token accuracy |
| `final.pt` | final-epoch checkpoint |
| `ckpt_epoch{N}.pt` | periodic checkpoints (every 5 epochs) |
| `metrics.json` | test metrics + training stats |
| `history.json` | per-epoch loss and accuracy |
| `samples.json` | sample predictions vs targets |
| `loss.png`, `accuracy.png` | training curves |

### Evaluation

To re-evaluate a trained checkpoint and produce length-matched metrics
(predictions scored against targets truncated to the same length, so models are not
penalised for sequence tails beyond the decode budget):

```bash
python src/samples.py
```

Set `CONFIG` and `CKPT` at the top of the file. Writes `metrics_length_matched.json`
and `samples_best.json` to the same output directory.

### Hyperparameters

Shared across all five configurations:

| Parameter | Value |
|---|---|
| model dim | 512 |
| attention heads | 8 |
| encoder / decoder layers | 6 |
| FFN expansion | 4× |
| sequence length | 320 |
| batch size | 32 |
| epochs | 20 |
| optimizer | AdamW, lr 3e-4 |
| gradient clipping | 1.0 |
| decoding | greedy |

C5 additionally uses 2 local encoder layers, 2 local decoder layers, and patch size 1.
C1–C4 use BPE tokenizers with a source vocabulary of 256 and target vocabulary of 512;
C5 is token-free with a shared byte vocabulary of 256.

---

## Results

Length-matched metrics on the test set, greedy decoding:

| Config | Bit Acc | Seq Acc | Levenshtein ↓ | Params | Time (min) | Peak VRAM (GB) |
|---|---|---|---|---|---|---|
| C1 (base) | 0.9181 | 0.001 | 36.01 | 44.8M | 18.4 | 9.22 |
| C2 (RoPE) | 0.6633 | 0.000 | 136.90 | 44.8M | 19.0 | 9.23 |
| C3 (GQA) | 0.9143 | 0.000 | 34.62 | 37.7M | 16.9 | 9.10 |
| C4 (RMSNorm) | 0.9167 | 0.004 | 35.66 | 44.8M | 17.8 | 8.46 |
| **C5 (BLT)** | **0.9957** | **0.813** | **1.10** | **36.4M** | **13.8** | **6.99** |

See `Report.pdf` for the full ablation analysis.

---

## Pretrained Checkpoints

All five checkpoints are hosted on Hugging Face:

**https://huggingface.co/ZappY-AI/anlp-a1**

| Config | Path |
|---|---|
| C1 | [`c1/best.pt`](https://huggingface.co/ZappY-AI/anlp-a1/tree/main/c1) |
| C2 | [`c2/best.pt`](https://huggingface.co/ZappY-AI/anlp-a1/tree/main/c2) |
| C3 | [`c3/best.pt`](https://huggingface.co/ZappY-AI/anlp-a1/tree/main/c3) |
| C4 | [`c4/best.pt`](https://huggingface.co/ZappY-AI/anlp-a1/tree/main/c4) |
| C5 | [`c5/best.pt`](https://huggingface.co/ZappY-AI/anlp-a1/tree/main/c5) |

Loading a checkpoint:

```python
import torch
from huggingface_hub import hf_hub_download
from models.architectures import build_model

path = hf_hub_download(repo_id="ZappY-AI/anlp-a1", filename="c5/best.pt")
model = build_model("C5", meta, 512, 8, 6, 1, 2, 2)
model.load_state_dict(torch.load(path, map_location="cpu"))
model.eval()
```

`meta` is returned by the dataloader constructors in `src/dataset.py` and carries the
vocabulary sizes and special-token IDs the model needs.

---

## Weights & Biases

Training runs for all five configurations are logged here:

**https://wandb.ai/mohd-zeeshan-iiit-hyderabad/anlp-assignment1/workspace**

Each run records per-step training loss, per-epoch train/validation loss and token
accuracy, peak GPU memory, and elapsed time.

---

## Project Structure

```
2025901020_assignment1/
├── src/
│   ├── models/
│   │   ├── attention.py        # MHA, GQA, and their cross-attention variants
│   │   ├── positional.py       # sinusoidal embeddings and RoPE
│   │   ├── norm.py             # LayerNorm and RMSNorm
│   │   ├── blt.py              # BLT local encoder/decoder and patch modules
│   │   └── architectures.py    # Transformer and BLT assembly, build_model
│   ├── bpe.py                  # BPE tokenizers for cipher and plaintext
│   ├── dataset.py              # tokenized and token-free dataloaders
│   ├── train.py                # training loop with W&B logging
│   ├── samples.py              # evaluation and sample generation
│   └── utils.py                # metrics, greedy decoding, plots
├── outputs/                    # checkpoints, metrics, logs, plots per config
├── Dataset_A1/                 # cipher and plaintext data
├── split_indices.json          # fixed train/val/test split
├── README.md
└── Report.pdf
```