import os
import time
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from bpe import CipherBPETokenizer, PlainTextTokenizer
from dataset import get_bpe_dataloaders, get_byte_dataloaders
from models.architectures import build_model
from utils import greedy_decode, remove_specials, token_accuracy, compute_all_metrics, plot_curves, save_json


CONFIG = "C5"

ROOT = r"C:\Users\hi\Desktop\ANLP Assignments\2025901020_assignment1"
CIPHER_PATH = os.path.join(ROOT, "Dataset_A1", "brown_cipher.txt")
PLAIN_PATH = os.path.join(ROOT, "Dataset_A1", "brown_plain.txt")
SPLIT_PATH = os.path.join(ROOT, "split_indices.json")
OUT_DIR = os.path.join(ROOT, "outputs", CONFIG.lower())

SEQ_LEN = 320
MAX_DECODE_LEN = 320 if CONFIG == "C5" else 192
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
MODEL_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
LOCAL_ENC_LAYERS = 2
LOCAL_DEC_LAYERS = 2
PATCH_SIZE = 1
CKPT_EVERY = 5
NUM_SAMPLES = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    if CONFIG == "C5":
        loaders = get_byte_dataloaders(CIPHER_PATH, PLAIN_PATH, SPLIT_PATH, SEQ_LEN, BATCH_SIZE, source_length=SEQ_LEN)
        return (*loaders, None)

    src_tok = CipherBPETokenizer(vocab_size=256)
    src_tok.load(os.path.join(ROOT, "tokenizer_cipher.json"))
    tgt_tok = PlainTextTokenizer(vocab_size=512)
    tgt_tok.load(os.path.join(ROOT, "tokenizer_plain.json"))

    train, val, test, meta = get_bpe_dataloaders(
        CIPHER_PATH, PLAIN_PATH, SPLIT_PATH, src_tok, tgt_tok, SEQ_LEN, BATCH_SIZE
    )
    return train, val, test, meta, tgt_tok


def ids_to_text(ids, specials, tokenizer):
    ids = remove_specials(ids, specials)
    if tokenizer is None:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")
    return tokenizer.decode(ids)


def run_epoch(model, loader, criterion, pad_id, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    total_loss, total_acc, n = 0.0, 0.0, 0
    for src, dec_in, dec_out in tqdm(loader, leave=False):
        src, dec_in, dec_out = src.to(device), dec_in.to(device), dec_out.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(src, dec_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_out.reshape(-1))

        if train_mode:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            wandb.log({"per_step_train_loss": loss.item()})

        total_loss += loss.item()
        total_acc += token_accuracy(logits, dec_out, pad_id)
        n += 1

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_test(model, loader, meta, tokenizer):
    specials = meta["target_specials"]
    preds, golds = [], []

    for src, _, dec_out in tqdm(loader, desc="decoding", leave=False):
        src, dec_out = src.to(device), dec_out.to(device)
        generated = greedy_decode(model, src, specials["sos_token_id"], specials["eos_token_id"], MAX_DECODE_LEN)

        for p, g in zip(generated, dec_out):
            preds.append(ids_to_text(p.tolist(), specials, tokenizer))
            golds.append(ids_to_text(g.tolist(), specials, tokenizer))

    return preds, golds


def main():
    train_loader, val_loader, test_loader, meta, tokenizer = load_data()
    pad_id = meta["target_specials"]["pad_token_id"]

    model = build_model(CONFIG, meta, MODEL_DIM, NUM_HEADS, NUM_LAYERS, PATCH_SIZE, LOCAL_ENC_LAYERS, LOCAL_DEC_LAYERS).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"{CONFIG} | {num_params/1e6:.2f}M params | device={device}")

    wandb.init(project="anlp-assignment1", name=CONFIG, config={
        "config": CONFIG, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE, "epochs": EPOCHS,
        "lr": LR, "model_dim": MODEL_DIM, "num_heads": NUM_HEADS, "num_layers": NUM_LAYERS,
        "patch_size": PATCH_SIZE, "num_params": num_params,
    })

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0   # to run only the best model for evaluation
    best_epoch = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, pad_id, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, pad_id)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
        print(f"epoch {epoch:02d} | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} | mem {peak_mem:.2f}GB")

        wandb.log({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "train_token_acc": tr_acc, "val_token_acc": va_acc,
            "peak_memory_gb": peak_mem, "elapsed_min": (time.time() - start) / 60,
        })
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "best.pt"))

        if epoch % CKPT_EVERY == 0 or epoch == EPOCHS:
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"ckpt_epoch{epoch}.pt"))

    train_time = time.time() - start
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final.pt"))
    plot_curves(history, OUT_DIR)

    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "best.pt"), map_location=device))
    print(f"evaluating best checkpoint (epoch {best_epoch}, val acc {best_val_acc:.4f})")

    preds, golds = evaluate_test(model, test_loader, meta, tokenizer)
    metrics = compute_all_metrics(preds, golds, tokenized=(CONFIG != "C5"))
    metrics.update({
        "train_time_min": train_time / 60,
        "peak_memory_gb": peak_mem,
        "num_params": num_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
    })

    print(json.dumps(metrics, indent=2) if False else metrics)
    wandb.log({f"test/{k}": v for k, v in metrics.items()})

    save_json(metrics, os.path.join(OUT_DIR, "metrics.json"))
    save_json(history, os.path.join(OUT_DIR, "history.json"))
    save_json(
        [{"prediction": p, "target": g} for p, g in zip(preds[:NUM_SAMPLES], golds[:NUM_SAMPLES])],
        os.path.join(OUT_DIR, "samples.json"),
    )

    wandb.finish()


if __name__ == "__main__":
    main()