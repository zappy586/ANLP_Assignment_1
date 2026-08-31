import json
import torch
import random
from tqdm import tqdm
from collections import Counter

class CipherBPETokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {0: "0", 1: "1"}
        self.merges = {}

    def get_stats(self, ids):
        return Counter(zip(ids, ids[1:]))

    def merge_ids(self, ids, pair, new_idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, lines):
        ids_list = [[int(b) for b in line] for line in lines]
        num_merges = self.vocab_size - 2

        for i in tqdm(range(num_merges)):
            stats = Counter()
            for ids in ids_list:
                stats.update(self.get_stats(ids))
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_idx = 2 + i

            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids_list = [self.merge_ids(ids, best_pair, new_idx) for ids in ids_list]

    def encode(self, text, max_len=None):
        if max_len is not None:
            text = text[: max_len * 20]
        ids = [int(b) for b in text]
        while len(ids) >= 2:
            if max_len is not None and len(ids) <= max_len:
                break
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self.merge_ids(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.vocab[idx] for idx in ids)

    def save(self, filepath):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": [[[p[0], p[1]], idx] for p, idx in self.merges.items()],
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)

    def load(self, filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.merges = {(p[0], p[1]): idx for p, idx in data["merges"]}


class PlainTextTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}

    def get_stats(self, ids):
        return Counter(zip(ids, ids[1:]))

    def merge_ids(self, ids, pair, new_idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, lines):
        ids_list = [list(line.encode("utf-8")) for line in lines]
        num_merges = self.vocab_size - 256

        for i in tqdm(range(num_merges)):
            stats = Counter()
            for ids in ids_list:
                stats.update(self.get_stats(ids))
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            new_idx = 256 + i

            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids_list = [self.merge_ids(ids, best_pair, new_idx) for ids in ids_list]

    def encode(self, text, max_len=None):
        if max_len is not None:
            text = text[: max_len * 20]
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            if max_len is not None and len(ids) <= max_len:
                break
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = self.merge_ids(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        byte_seq = b"".join(self.vocab[idx] for idx in ids)
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, filepath):
        data = {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v.decode("latin-1") for k, v in self.vocab.items()},
            "merges": [[[p[0], p[1]], idx] for p, idx in self.merges.items()],
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.vocab = {int(k): v.encode("latin-1") for k, v in data["vocab"].items()}
        self.merges = {(p[0], p[1]): idx for p, idx in data["merges"]}



def main():
    with open(r"C:\Users\hi\Desktop\ANLP Assignments\2025901020_assignment1\Dataset_A1\brown_cipher.txt", "r", encoding="utf-8") as f:
        cipher_lines = [line.strip() for line in f]

    with open(r"C:\Users\hi\Desktop\ANLP Assignments\2025901020_assignment1\Dataset_A1\brown_plain.txt", "r", encoding="utf-8") as f:
        plain_lines = [line.strip() for line in f]

    # splitting into train/test to avoid test split's dataset statistics from leaking into training
    indices = list(range(len(cipher_lines)))
    random.seed(42)
    random.shuffle(indices)

    split_idx = int(len(indices) * 0.7)
    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

    with open("split_indices.json", "w", encoding="utf-8") as f:
        json.dump({"train": train_idx, "test": test_idx}, f)

    cipher_train = [cipher_lines[i] for i in train_idx]
    plain_train = [plain_lines[i] for i in train_idx]

    cipher_tok = CipherBPETokenizer(vocab_size=256)
    cipher_tok.train(cipher_train)
    cipher_tok.save("tokenizer_cipher.json")

    plain_tok = PlainTextTokenizer(vocab_size=512)
    plain_tok.train(plain_train)
    plain_tok.save("tokenizer_plain.json")

if __name__ == "__main__":
    main()