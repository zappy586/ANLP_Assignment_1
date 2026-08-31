import os
import json
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Levenshtein
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer


@torch.no_grad()
def greedy_decode(model, src, sos_id, eos_id, max_len):
    model.eval()
    b = src.shape[0]
    enc, src_m = model.encode(src)

    tgt = torch.full((b, 1), sos_id, dtype=torch.long, device=src.device)
    done = torch.zeros(b, dtype=torch.bool, device=src.device)

    for _ in range(max_len - 1):
        logits = model.decode(tgt, enc, src_m)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        nxt[done] = eos_id
        tgt = torch.cat([tgt, nxt], dim=1)
        done |= nxt.squeeze(1) == eos_id
        if done.all():
            break
    return tgt

def remove_specials(ids, specials):
    out = []
    for i in ids:
        i = int(i)
        if i == specials["eos_token_id"]:
            break
        if i in (specials["sos_token_id"], specials["pad_token_id"]):
            continue
        out.append(i)
    return out


def token_accuracy(logits, targets, pad_id):
    preds = logits.argmax(-1)
    mask = targets != pad_id
    if mask.sum() == 0:
        return 0.0
    return ((preds == targets) & mask).sum().item() / mask.sum().item()


def bit_accuracy(pred_strs, gold_strs):
    total, correct = 0, 0
    for p, g in zip(pred_strs, gold_strs):
        pb = "".join(f"{c:08b}" for c in p.encode("utf-8", errors="replace"))
        gb = "".join(f"{c:08b}" for c in g.encode("utf-8", errors="replace"))
        n = min(len(pb), len(gb))
        correct += sum(a == b for a, b in zip(pb[:n], gb[:n]))
        total += max(len(pb), len(gb))
    return correct / total if total else 0.0


def sequence_accuracy(pred_strs, gold_strs):
    if not pred_strs:
        return 0.0
    return sum(p == g for p, g in zip(pred_strs, gold_strs)) / len(pred_strs)


def avg_levenshtein(pred_strs, gold_strs):
    if not pred_strs:
        return 0.0
    return sum(Levenshtein.distance(p, g) for p, g in zip(pred_strs, gold_strs)) / len(pred_strs)


def bleu_score(pred_strs, gold_strs):
    refs = [[g.split()] for g in gold_strs]
    hyps = [p.split() for p in pred_strs]
    return corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method1)


def rouge_scores(pred_strs, gold_strs):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    r1, rl = [], []
    for p, g in zip(pred_strs, gold_strs):
        s = scorer.score(g, p)
        r1.append(s["rouge1"].fmeasure)
        rl.append(s["rougeL"].fmeasure)
    return (sum(r1) / len(r1) if r1 else 0.0), (sum(rl) / len(rl) if rl else 0.0)


def compute_all_metrics(pred_strs, gold_strs, tokenized=True):
    metrics = {
        "bit_accuracy": bit_accuracy(pred_strs, gold_strs),
        "sequence_accuracy": sequence_accuracy(pred_strs, gold_strs),
        "levenshtein": avg_levenshtein(pred_strs, gold_strs),
    }
    if tokenized:
        metrics["bleu"] = bleu_score(pred_strs, gold_strs)
        r1, rl = rouge_scores(pred_strs, gold_strs)
        metrics["rouge1"] = r1
        metrics["rougeL"] = rl
    return metrics


def plot_curves(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=120)
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("token accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "accuracy.png"), dpi=120)
    plt.close()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)