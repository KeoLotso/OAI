import argparse
import os
import math
import json
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.seq_len = seq_len
        self.tokens = tokens
        self.n = len(tokens) - seq_len
    def __len__(self):
        return max(0, self.n)
    def __getitem__(self, idx):
        return torch.tensor(self.tokens[idx:idx+self.seq_len], dtype=torch.long), torch.tensor(self.tokens[idx+1:idx+self.seq_len+1], dtype=torch.long)

class SmallLM(nn.Module):
    def __init__(self, vocab_size, emb=128, hid=256, n_layers=2, dropout=0.2):
        super().__init__()
        self.e = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hid, n_layers, dropout=dropout, batch_first=True)
        self.lin = nn.Linear(hid, vocab_size)
    def forward(self, x, hidden=None):
        x = self.e(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.lin(out)
        return logits, hidden
    def init_hidden(self, bsz, device):
        n_layers = self.lstm.num_layers
        hid = self.lstm.hidden_size
        return (torch.zeros(n_layers, bsz, hid, device=device), torch.zeros(n_layers, bsz, hid, device=device))

def build_vocab(text, vocab_size=10000, min_freq=1):
    words = text.split()
    freqs = Counter(words)
    common = [w for w,c in freqs.most_common(vocab_size) if c>=min_freq]
    itos = ["<pad>", "<unk>"] + common
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

def encode(text, stoi):
    return [stoi.get(w, stoi["<unk>"]) for w in text.split()] #unc

def decode(indices, itos):
    return " ".join(itos[i] if i < len(itos) else "<unk>" for i in indices)

def top_k_sample(probs, k):
    if k == 0:
        return torch.multinomial(probs, 1).item()
    vals, idx = torch.topk(probs, k)
    vals = vals / vals.sum()
    return idx[torch.multinomial(vals, 1).item()].item()

def generate(model, stoi, itos, seed_text, length=50, temperature=1.0, top_k=40, device="cpu"):
    model.eval()
    tokens = seed_text.split()
    cur = [stoi.get(w, stoi["<unk>"]) for w in tokens]
    seq = cur.copy()
    ctx = 20
    input_seq = torch.tensor([cur[-ctx:]], dtype=torch.long, device=device)
    hidden = None
    with torch.no_grad():
        logits, hidden = model(input_seq, None)
        last = logits[0, -1]
        for _ in range(length):
            logits_scaled = last / max(1e-8, temperature)
            probs = torch.softmax(logits_scaled, dim=-1)
            idx = top_k_sample(probs, top_k)
            seq.append(idx)
            inp = torch.tensor([[idx]], dtype=torch.long, device=device)
            logits, hidden = model(inp, hidden)
            last = logits[0, -1]
    return decode(seq, itos)

def save_state(path, model, stoi, itos, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    meta = {"stoi": stoi, "itos": itos, "args": vars(args)}
    with open(path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

def load_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys

def train(args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    text = load_text(args.data)
    stoi, itos = build_vocab(text, vocab_size=args.vocab_size, min_freq=args.min_freq)
    tokens = encode(text, stoi)
    dataset = TextDataset(tokens, args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = SmallLM(len(itos), emb=args.emb_size, hid=args.hidden_size, n_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for xb, yb in pbar:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits, _ = model(xb, None)
            b, s, v = logits.shape
            loss = criterion(logits.view(b*s, v), yb.view(b*s))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=total_loss / (pbar.n+1))
        ppl = math.exp(total_loss / max(1, len(loader)))
        print(f"Epoch {epoch} finished. PPL: {ppl:.2f}")
        save_state(args.save_path, model, stoi, itos, args)
        sample_seed = args.sample_seed if args.sample_seed else "The"
        print("SAMPLE:", generate(model, stoi, itos, sample_seed, length=50, temperature=args.temp, top_k=args.top_k, device=device)) #most likely the most worst thing ever bruh
    print("Training done. Model saved to", args.save_path)

def interact(model_path, args):
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    meta_path = model_path + ".meta.json"
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise SystemExit("Model or meta not found.")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]
    model = SmallLM(len(itos), emb=args.emb_size, hid=args.hidden_size, n_layers=args.layers, dropout=args.dropout).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    while True:
        seed = input(">>> ").strip()
        if seed.lower() in ("quit", "exit"):
            break
        out = generate(model, stoi, itos, seed if seed else args.default_prompt, length=args.gen_len, temperature=args.temp, top_k=args.top_k, device=device)
        print(out)

def main():
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)
    default_data = os.path.join(base_dir, "data.txt")

    choice = input("Do you want to [t]rain or [i]nteract? ").strip().lower() #whooo smart boy

    if choice.startswith("t"):
        model_name = input("Enter model name: ").strip()
        if not model_name:
            model_name = "default_model"
        model_path = os.path.join(models_dir, f"{model_name}.pt")
        args = argparse.Namespace(
            data=default_data,
            save_path=model_path,
            vocab_size=8000,
            min_freq=1,
            seq_len=30,
            emb_size=128,
            hidden_size=256,
            layers=2,
            dropout=0.2,
            batch_size=64,
            epochs=6,
            lr=1e-3,
            no_cuda=False,
            sample_seed="",
            temp=1.0,
            top_k=40,
            gen_len=50,
            default_prompt="Hello",
            mode="train"
        )
        train(args)
    else:
        models = [f for f in os.listdir(models_dir) if f.endswith(".pt")]
        if not models:
            print("No models found.")
            return
        print("Available models:")
        for i, m in enumerate(models, 1):
            print(f"{i}. {m}")
        choice = input("Select model number: ").strip()
        try:
            idx = int(choice) - 1
            model_file = models[idx]
        except:
            print("Invalid")
            return
        model_path = os.path.join(models_dir, model_file)
        args = argparse.Namespace(
            emb_size=128,
            hidden_size=256,
            layers=2,
            dropout=0.2,
            no_cuda=False,
            temp=1.0,
            top_k=40,
            gen_len=50,
            default_prompt="Hello",
            mode="interact"
        )
        interact(model_path, args)

if __name__ == "__main__":
    main()
