import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def make_special_ids(vocab_size):
    return {
        "pad_token_id": vocab_size,
        "sos_token_id": vocab_size + 1,
        "eos_token_id": vocab_size + 2,
        "embedding_size": vocab_size + 3,
    }


class CipherDataset(Dataset):
    def __init__(self, source_sequences, target_sequences, sequence_length, source_specials, target_specials, source_length=None):
        self.source_sequences = source_sequences
        self.target_sequences = target_sequences
        self.sequence_length = sequence_length
        self.source_length = source_length or sequence_length
        self.source_specials = source_specials
        self.target_specials = target_specials

    def __len__(self):
        return len(self.source_sequences)

    def format_sequence(self, sequence, specials, length):
        sequence = sequence[: length - 2]
        formatted = [specials["sos_token_id"]] + list(sequence) + [specials["eos_token_id"]]
        formatted.extend([specials["pad_token_id"]] * (length - len(formatted)))
        return formatted

    def __getitem__(self, index):
        source = self.format_sequence(self.source_sequences[index], self.source_specials, self.source_length)
        target = self.format_sequence(self.target_sequences[index], self.target_specials, self.sequence_length)

        encoder_input = torch.tensor(source, dtype=torch.long)
        decoder_sequence = torch.tensor(target, dtype=torch.long)

        return encoder_input, decoder_sequence[:-1], decoder_sequence[1:]


def load_lines(cipher_file_path, plaintext_file_path, max_plain_chars=256):
    max_cipher_bits = max_plain_chars * 8
    with open(cipher_file_path, "r", encoding="utf-8") as f:
        cipher_lines = [line.strip()[:max_cipher_bits] for line in f]
    with open(plaintext_file_path, "r", encoding="utf-8") as f:
        plain_lines = [line.strip()[:max_plain_chars] for line in f]
    assert len(cipher_lines) == len(plain_lines), f"line count mismatch: {len(cipher_lines)} vs {len(plain_lines)}"
    return cipher_lines, plain_lines

def load_splits(split_path, val_fraction_of_holdout=0.5):
    with open(split_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    train_idx = splits["train"]
    held_out = splits["test"]

    val_end = int(len(held_out) * val_fraction_of_holdout)
    return train_idx, held_out[:val_end], held_out[val_end:]


def build_loader(sources, targets, sequence_length, batch_size, source_specials, target_specials, shuffle_data, source_length=None):
    dataset = CipherDataset(sources, targets, sequence_length, source_specials, target_specials, source_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_data)


def get_bpe_dataloaders(cipher_file_path, plaintext_file_path, split_path, source_tokenizer, target_tokenizer, sequence_length, batch_size, cache_path="bpe_encoded_tokens_cache.json"):
    cipher_lines, plain_lines = load_lines(cipher_file_path, plaintext_file_path)
    train_idx, val_idx, test_idx = load_splits(split_path)
    source_specials = make_special_ids(source_tokenizer.vocab_size)
    target_specials = make_special_ids(target_tokenizer.vocab_size)
    max_len = sequence_length - 2
    cache_key = {
        "sequence_length": sequence_length,
        "source_vocab_size": source_tokenizer.vocab_size,
        "target_vocab_size": target_tokenizer.vocab_size,
        "source_merges": len(source_tokenizer.merges),
        "target_merges": len(target_tokenizer.merges),
        "num_lines": len(cipher_lines),
    }
    cached = None
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            candidate = json.load(f)
        if candidate.get("key") == cache_key:
            cached = candidate
            print(f"loaded encodings from {cache_path}")
        else:
            print("cache key mismatch, re-encoding")
    if cached is None:
        encoded_sources = {}
        encoded_targets = {}
        for i in tqdm(train_idx + val_idx + test_idx, desc="encoding"):
            encoded_sources[i] = source_tokenizer.encode(cipher_lines[i], max_len=max_len)
            encoded_targets[i] = target_tokenizer.encode(plain_lines[i], max_len=max_len)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "key": cache_key,
                "sources": {str(k): v for k, v in encoded_sources.items()},
                "targets": {str(k): v for k, v in encoded_targets.items()},
            }, f)
        print(f"saved encodings to {cache_path}")
    else:
        encoded_sources = {int(k): v for k, v in cached["sources"].items()}
        encoded_targets = {int(k): v for k, v in cached["targets"].items()}
    def make(idx_list, shuffle_data):
        sources = [encoded_sources[i] for i in idx_list]
        targets = [encoded_targets[i] for i in idx_list]
        return build_loader(sources, targets, sequence_length, batch_size, source_specials, target_specials, shuffle_data)
    meta = {
        "source_specials": source_specials,
        "target_specials": target_specials,
        "source_vocab_size": source_tokenizer.vocab_size,
        "target_vocab_size": target_tokenizer.vocab_size,
    }
    return make(train_idx, True), make(val_idx, False), make(test_idx, False), meta


def get_byte_dataloaders(cipher_file_path, plaintext_file_path, split_path, sequence_length, batch_size, byte_vocab_size=256, source_length=None):
    cipher_lines, plain_lines = load_lines(cipher_file_path, plaintext_file_path)
    train_idx, val_idx, test_idx = load_splits(split_path)

    specials = make_special_ids(byte_vocab_size)

    def make(idx_list, shuffle_data):
        sources = [list(cipher_lines[i].encode("utf-8")) for i in idx_list]
        targets = [list(plain_lines[i].encode("utf-8")) for i in idx_list]
        return build_loader(sources, targets, sequence_length, batch_size, specials, specials, shuffle_data, source_length)

    meta = {
        "source_specials": specials,
        "target_specials": specials,
        "source_vocab_size": byte_vocab_size,
        "target_vocab_size": byte_vocab_size,
    }

    return make(train_idx, True), make(val_idx, False), make(test_idx, False), meta

