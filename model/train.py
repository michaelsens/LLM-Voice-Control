# train_lstm.py
# Usage:
# python train_lstm.py --data ../expanded.jsonl --block_size 128 --batch 16 --epochs 20 --lr 3e-4 --val_ratio 0.1

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
from tqdm import tqdm
import os

def split_action_target(utt: str):
    parts = utt.strip().split(maxsplit=1)
    action = parts[0].lower()
    target = parts[1].strip() if len(parts) > 1 else ""
    return action, target

class Seq2SeqDataset(Dataset):
    def __init__(self, examples, enc, block_size):
        BOS, EOS = enc.n_vocab, enc.n_vocab + 1
        self.data = []
        for utt, rpc in examples:
            action, target = split_action_target(utt)
            inp_str = f"action: {action} target: {target}"
            src_ids = enc.encode(inp_str)[:block_size]
            tgt_ids = enc.encode(json.dumps(rpc, separators=(',',':')))[:block_size-1]
            if not tgt_ids:
                continue
            self.data.append((
                torch.tensor([BOS] + src_ids, dtype=torch.long),
                torch.tensor([BOS] + tgt_ids, dtype=torch.long),
                torch.tensor(tgt_ids + [EOS], dtype=torch.long)
            ))
        if not self.data:
            raise ValueError("No valid examples found in dataset.")

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    src, inp, out = zip(*batch)
    max_s, max_t = max(map(len, src)), max(map(len, inp))
    B = len(src)
    src_pad = torch.zeros(max_s, B, dtype=torch.long)
    inp_pad = torch.zeros(max_t, B, dtype=torch.long)
    out_pad = torch.zeros(max_t, B, dtype=torch.long)
    for i, (s, ii, oo) in enumerate(batch):
        src_pad[:len(s), i] = s
        inp_pad[:len(ii), i] = ii
        out_pad[:len(oo), i] = oo
    src_mask = (src_pad==0).transpose(0,1)
    tgt_mask = (inp_pad==0).transpose(0,1)
    return src_pad, inp_pad, out_pad, src_mask, tgt_mask

class Seq2SeqLSTM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.LSTM(d_model, d_model, num_layers=num_layers, dropout=dropout)
        self.decoder = nn.LSTM(d_model, d_model, num_layers=num_layers, dropout=dropout)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        embedded_src = self.embedding(src)
        _, (hidden, cell) = self.encoder(embedded_src)
        embedded_tgt = self.embedding(tgt)
        outputs, _ = self.decoder(embedded_tgt, (hidden, cell))
        return self.fc_out(outputs)

def train(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for src, inp, out, src_mask, tgt_mask in tqdm(loader, desc="Training"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        optimizer.zero_grad()
        logits = model(src, inp)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), out.view(-1), ignore_index=0, label_smoothing=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def eval_token_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for src, inp, out, src_mask, tgt_mask in tqdm(loader, desc="Evaluating"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        logits = model(src, inp)
        preds = logits.argmax(dim=-1)
        mask = out.ne(0)
        correct += (preds == out).masked_select(mask).sum().item()
        total += mask.sum().item()
    return 100 * correct / total if total else 0.0

@torch.no_grad()
def infer(model, enc, device, utterance, block_size):
    src_ids = enc.encode(utterance)[:block_size]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device)
    BOS_ID = enc.n_vocab
    EOS_ID = enc.n_vocab + 1
    ys = torch.tensor([BOS_ID], dtype=torch.long).unsqueeze(1).to(device)
    for _ in range(block_size):
        logits = model(src, ys)
        next_id = logits[-1,0].argmax().unsqueeze(0).unsqueeze(1)
        ys = torch.cat([ys, next_id], dim=0)
        if next_id.item() == EOS_ID:
            break
    token_ids = ys.squeeze().tolist()
    return enc.decode(token_ids[1:-1])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--enc_layers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.1)
    args = parser.parse_args()

    if args.device=='cpu' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    examples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            examples.append((j['utterance'], j['rpc']))

    enc = tiktoken.get_encoding('gpt2')
    dataset = Seq2SeqDataset(examples, enc, args.block_size)
    total, val_size = len(dataset), int(args.val_ratio * len(dataset))
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,collate_fn=collate_fn, num_workers=num_workers)

    vocab_size = enc.n_vocab + 2
    model = Seq2SeqLSTM(vocab_size, d_model=args.d_model, num_layers=args.enc_layers, dropout=args.dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1)

    best_val, patience, no_imp = float('inf'), 3, 0
    for ep in range(1, args.epochs+1):
        trl = train(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {ep}/{args.epochs} — Train Loss: {trl:.4f}")
        total_loss = 0
        for src, inp, out, sm, tm in val_loader:
            logits = model(src.to(device), inp.to(device))
            total_loss += F.cross_entropy(logits.view(-1, vocab_size), out.to(device).view(-1), ignore_index=0, label_smoothing=0.1).item()
        vl = total_loss / len(val_loader)
        va = eval_token_accuracy(model, val_loader, device)
        print(f" -> Val Loss: {vl:.4f} — Val Acc: {va:.1f}%")
        if vl < best_val - 1e-4:
            best_val, no_imp = vl, 0
            torch.save(model.state_dict(), 'seq2seq_lstm.pt')
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break
    print("Training complete.")

if __name__ == '__main__':
    main()
