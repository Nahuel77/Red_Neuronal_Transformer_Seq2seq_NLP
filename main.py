# main.py
import json
import math
import re
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import Dataset, DataLoader

# -----------------------
# Config / Device
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SRC_LEN = 128
MAX_TGT_LEN = 40
BATCH_SIZE = 16
EPOCHS = 12
LR = 3e-4
D_MODEL = 128
NHEAD = 4
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
DIM_FF = 512
DROPOUT = 0.1
MODEL_PATH = "transformer_summarizer.pt"

# -----------------------
# Simple tokenizer
# -----------------------
def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúüñ \t]+", "", text)
    return text.strip().split()

def build_vocab(texts, max_size=20000, min_freq=1):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    items = [w for w, c in counter.most_common() if c >= min_freq]
    items = items[: max_size - 4]
    itos = ["<pad>", "<unk>", "<sos>", "<eos>"] + items
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos

def encode_tokens(tokens, stoi, max_len, add_specials=True):
    ids = []
    if add_specials:
        ids.append(stoi["<sos>"])
    for t in tokens:
        ids.append(stoi.get(t, stoi["<unk>"]))
        if len(ids) >= max_len - 1:
            break
    if add_specials:
        ids.append(stoi["<eos>"])
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    return ids[:max_len]

def encode_text(text, stoi, max_len):
    toks = tokenize(text)
    return encode_tokens(toks, stoi, max_len)

# -----------------------
# Dataset
# -----------------------
class NewsSummaryDataset(Dataset):
    def __init__(self, items, src_stoi, tgt_stoi):
        self.items = items
        self.src_stoi = src_stoi
        self.tgt_stoi = tgt_stoi

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        src_ids = encode_text(it["text"], self.src_stoi, MAX_SRC_LEN)
        tgt_ids = encode_text(it["summary"], self.tgt_stoi, MAX_TGT_LEN)
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

# -----------------------
# Positional Encoding
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# -----------------------
# Transformer
# -----------------------
class TransformerSummarizer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab):
        super().__init__()
        self.src_emb = nn.Embedding(src_vocab, D_MODEL, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab, D_MODEL, padding_idx=0)
        self.pos = PositionalEncoding(D_MODEL)

        self.tr = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FF,
            dropout=DROPOUT,
            batch_first=True
        )

        self.fc = nn.Linear(D_MODEL, tgt_vocab)

    def make_masks(self, src, tgt):
        src_pad = (src == 0)
        tgt_pad = (tgt == 0)

        L = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(L, L, device=src.device), diagonal=1).bool()
        return src_pad, tgt_pad, tgt_mask

    def forward(self, src, tgt):
        src = self.pos(self.src_emb(src))
        tgt = self.pos(self.tgt_emb(tgt))

        src_pad, tgt_pad, tgt_mask = self.make_masks(src.argmax(-1), tgt.argmax(-1))

        out = self.tr(
            src, tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_pad,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=src_pad
        )
        return self.fc(out)

    def encode(self, src):
        src = self.pos(self.src_emb(src))
        src_pad = (src.argmax(-1) == 0)
        mem = self.tr.encoder(src, src_key_padding_mask=src_pad)
        return mem, src_pad

    def decode_step(self, tgt, mem, mem_pad):
        tgt = self.pos(self.tgt_emb(tgt))
        L = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(L, L, device=tgt.device), 1).bool()

        out = self.tr.decoder(
            tgt, mem,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=mem_pad
        )
        return self.fc(out)

# -----------------------
# Greedy decode
# -----------------------
def greedy_decode(model, src_ids, tgt_stoi, tgt_itos):
    model.eval()
    src = torch.tensor(src_ids).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mem, mem_pad = model.encode(src)

        ys = torch.tensor([[tgt_stoi["<sos>"]]], device=DEVICE)

        for _ in range(MAX_TGT_LEN - 1):
            out = model.decode_step(ys, mem, mem_pad)
            next_id = out[:, -1].argmax(-1).item()
            ys = torch.cat([ys, torch.tensor([[next_id]], device=DEVICE)], dim=1)
            if next_id == tgt_stoi["<eos>"]:
                break

    decoded = []
    for i in ys[0].tolist():
        if i == tgt_stoi["<sos>"]:
            continue
        if i == tgt_stoi["<eos>"]:
            break
        decoded.append(tgt_itos[i])
    return " ".join(decoded)

# -----------------------
# Main
# -----------------------
def main():
    with open("dataset.json", "r", encoding="utf8") as f:
        data = json.load(f)

    split = int(0.8 * len(data))
    train_items = data[:split]
    test_items = data[split:]

    src_texts = [x["text"] for x in train_items]
    tgt_texts = [x["summary"] for x in train_items]

    src_stoi, src_itos = build_vocab(src_texts)
    tgt_stoi, tgt_itos = build_vocab(tgt_texts)

    train_ds = NewsSummaryDataset(train_items, src_stoi, tgt_stoi)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = TransformerSummarizer(len(src_itos), len(tgt_itos)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_stoi["<pad>"])

    # -----------------------
    # LOAD CHECKPOINT IF EXISTS
    # -----------------------
    if os.path.exists(MODEL_PATH):
        ck = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(ck["model_state"])
        print("Loaded existing model weights.")

    print("Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        for src, tgt in loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            inp = tgt[:, :-1]
            lbl = tgt[:, 1:]

            optimizer.zero_grad()
            out = model(src, inp)
            loss = criterion(out.reshape(-1, out.size(-1)), lbl.reshape(-1))
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {epoch}/{EPOCHS} - Loss: {total/len(loader):.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "src_stoi": src_stoi, "src_itos": src_itos,
        "tgt_stoi": tgt_stoi, "tgt_itos": tgt_itos
    }, MODEL_PATH)
    print("Saved model.")

    print("\n=== EXAMPLES ===\n")
    examples = test_items[:5]
    for ex in examples:
        enc = encode_text(ex["text"], src_stoi, MAX_SRC_LEN)
        pred = greedy_decode(model, enc, tgt_stoi, tgt_itos)
        print("Texto:", ex["text"])
        print("Real:", ex["summary"])
        print("Gen.:", pred)
        print()

# -----------------------
if __name__ == "__main__":
    main()