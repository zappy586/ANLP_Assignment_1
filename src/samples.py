import os
import torch

from tqdm import tqdm

from bpe import CipherBPETokenizer, PlainTextTokenizer
from dataset import get_bpe_dataloaders, get_byte_dataloaders
from models.architectures import build_model
from utils import greedy_decode, remove_specials, compute_all_metrics, save_json

CONFIG = "C1"
CKPT = "best.pt"
NUM_SAMPLES = 20
COMPUTE_METRICS = True

ROOT = r"C:\Users\hi\Desktop\ANLP Assignments\2025901020_assignment1"
CIPHER_PATH = os.path.join(ROOT, "Dataset_A1", "brown_cipher.txt")
PLAIN_PATH = os.path.join(ROOT, "Dataset_A1", "brown_plain.txt")
SPLIT_PATH = os.path.join(ROOT, "split_indices.json")
OUT_DIR = os.path.join(ROOT, "outputs", CONFIG.lower())

SEQ_LEN = 320
BATCH_SIZE = 32
MODEL_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
PATCH_SIZE = 8
MAX_DECODE_LEN = 320 if CONFIG == "C5" else 192

device = "cuda" if torch.cuda.is_available() else "cpu"

if CONFIG == "C5":
    _, _, test_loader, meta = get_byte_dataloaders(CIPHER_PATH, PLAIN_PATH, SPLIT_PATH, SEQ_LEN, BATCH_SIZE, source_length=2064)
    tokenizer = None
else:
    src_tok = CipherBPETokenizer(vocab_size=256)
    src_tok.load(os.path.join(ROOT, "tokenizer_cipher.json"))
    tokenizer = PlainTextTokenizer(vocab_size=512)
    tokenizer.load(os.path.join(ROOT, "tokenizer_plain.json"))
    _, _, test_loader, meta = get_bpe_dataloaders(CIPHER_PATH, PLAIN_PATH, SPLIT_PATH, src_tok, tokenizer, SEQ_LEN, BATCH_SIZE)

model = build_model(CONFIG, meta, MODEL_DIM, NUM_HEADS, NUM_LAYERS, PATCH_SIZE, 4, 4).to(device)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, CKPT), map_location=device))
model.eval()
# # --- sanity check, delete after ---
# PAD = meta["target_specials"]["pad_token_id"]
# src, tgt_in, tgt_out = next(iter(test_loader))
# src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

# with torch.no_grad():
#     enc, cross_m = model.encode(src)

# n_valid = cross_m.sum(-1).squeeze()[0].item()
# print("patches:", enc.shape[1], "| valid:", n_valid,
#       "| target tokens:", (tgt_out[0] != PAD).sum().item())

# e = enc[0, :n_valid]
# sim = torch.nn.functional.cosine_similarity(e.unsqueeze(1), e.unsqueeze(0), dim=-1)
# print("mean pairwise cos sim:", sim.mean().item())
# print("std across patches:", e.std(0).mean().item())

# with torch.no_grad():
#     logits_real = model(src, tgt_in)
#     enc, cross_m = model.encode(src)
#     logits_shuf = model.decode(tgt_in, enc[torch.randperm(enc.shape[0])], cross_m)
# print("mean abs logit diff:", (logits_real - logits_shuf).abs().mean().item())
# --- end ---
print(f"loaded {CKPT} for {CONFIG}")

specials = meta["target_specials"]


def ids_to_text(ids):
    ids = remove_specials(ids, specials)
    if tokenizer is None:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")
    return tokenizer.decode(ids)


preds, golds = [], []
with torch.no_grad():
    for src, _, dec_out in tqdm(test_loader):
        src, dec_out = src.to(device), dec_out.to(device)
        gen = greedy_decode(model, src, specials["sos_token_id"], specials["eos_token_id"], MAX_DECODE_LEN)

        for p, g in zip(gen, dec_out):
            preds.append(ids_to_text(p.tolist()))
            golds.append(ids_to_text(g.tolist()))

        if not COMPUTE_METRICS and len(preds) >= NUM_SAMPLES:
            break

for i, (p, g) in enumerate(zip(preds[:NUM_SAMPLES], golds[:NUM_SAMPLES])):
    print(f"\n--- {i} ---")
    print(f"pred: {p}")
    print(f"gold: {g}")

tag = CKPT.replace(".pt", "")
save_json(
    [{"prediction": p, "target": g} for p, g in zip(preds[:NUM_SAMPLES], golds[:NUM_SAMPLES])],
    os.path.join(OUT_DIR, f"samples_{tag}.json"),
)

if COMPUTE_METRICS:
    metrics_raw = compute_all_metrics(preds, golds, tokenized=(CONFIG != "C5"))
    print("raw:", metrics_raw)

    golds_trunc = [g[:len(p)] for p, g in zip(preds, golds)]
    metrics_trunc = compute_all_metrics(preds, golds_trunc, tokenized=(CONFIG != "C5"))
    print("length-matched:", metrics_trunc)

    save_json({"raw": metrics_raw, "length_matched": metrics_trunc},
              os.path.join(OUT_DIR, f"metrics_{tag}.json"))