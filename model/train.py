# train.py
# Usage:
# python train.py --data ../expanded.jsonl --block_size 128 --batch 16 --epochs 20 --lr 3e-4 --val_ratio 0.1

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

# ----------------------------------------------------------------------------
# Utterance parser: split first verb as action, rest as target
# ----------------------------------------------------------------------------
def split_action_target(utt: str):
    parts = utt.strip().split(maxsplit=1)
    action = parts[0].lower()
    target = parts[1].strip() if len(parts) > 1 else ""
    return action, target

# ----------------------------------------------------------------------------
# Positional embeddings (Rotary)
# ----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10.0):
        super().__init__()
        inv_freq = 1.0 / (max_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        # return [seq_len, dim]
        return emb

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Collate
# ----------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------
# Model with true cross-attention
# ----------------------------------------------------------------------------
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.pos = RotaryEmbedding(d_model)

        # use default batch_first=False
        enc_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(dec_layer, num_decoder_layers)

        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src/tgt: [seq_len, batch]
        src_emb = self.src_emb(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_emb(tgt) * math.sqrt(self.d_model)

        # positional: [seq_len, d_model]
        Ls, Lt = src.size(0), tgt.size(0)
        pos_s = self.pos(Ls, src.device).unsqueeze(1)       # [seq_len, 1, d_model]
        pos_t = self.pos(Lt, tgt.device).unsqueeze(1)       # [seq_len, 1, d_model]
        src_emb = src_emb + pos_s
        tgt_emb = tgt_emb + pos_t

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(tgt_emb,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
        return self.generator(out)

# ----------------------------------------------------------------------------
# Training & evaluation
# ----------------------------------------------------------------------------
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


def train(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for src, inp, out, src_mask, tgt_mask in tqdm(loader, desc="Training"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        optimizer.zero_grad()
        logits = model(src, inp,
                       src_key_padding_mask=src_mask,
                       tgt_key_padding_mask=tgt_mask,
                       tgt_mask=generate_square_subsequent_mask(inp.size(0)).to(device))
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
        logits = model(src, inp,
                       src_key_padding_mask=src_mask,
                       tgt_key_padding_mask=tgt_mask,
                       tgt_mask=generate_square_subsequent_mask(inp.size(0)).to(device))
        preds = logits.argmax(dim=-1)
        mask = out.ne(0)
        correct += (preds == out).masked_select(mask).sum().item()
        total += mask.sum().item()
    return 100 * correct / total if total else 0.0


@torch.no_grad()
def infer(model, enc, device, utterance, block_size):
    # 1) Tokenize and build source tensor
    src_ids = enc.encode(utterance)[:block_size]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device)  # [seq,1]

    # 2) Start with BOS token
    BOS_ID = enc.n_vocab
    EOS_ID = enc.n_vocab + 1
    ys = torch.tensor([BOS_ID], dtype=torch.long).unsqueeze(1).to(device)

    # 3) Greedily generate up to block_size steps
    for _ in range(block_size):
        logits = model(src, ys)              # [tgt_len,1,vocab]
        next_id = logits[-1,0].argmax().unsqueeze(0).unsqueeze(1)
        ys = torch.cat([ys, next_id], dim=0)
        if next_id.item() == EOS_ID:
            break

    # 4) Decode (skip BOS/EOS)
    token_ids = ys.squeeze().tolist()
    return enc.decode(token_ids[1:-1])

# ----------------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()

    # select device
    if args.device=='cpu' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')

    # load data
    examples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            examples.append((j['utterance'], j['rpc']))

    # prepare dataset & loaders
    enc = tiktoken.get_encoding('gpt2')
    dataset = Seq2SeqDataset(examples, enc, args.block_size)
    total, val_size = len(dataset), int(args.val_ratio * len(dataset))
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    num_workers = min(8, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,collate_fn=collate_fn, num_workers=num_workers)

    # init model
    vocab_size = enc.n_vocab + 2
    model = Seq2SeqTransformer(vocab_size, args.d_model, args.nhead, args.enc_layers, args.dec_layers, args.ff, args.dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1)

    # training loop...
    best_val, patience, no_imp = float('inf'), 3, 0
    for ep in range(1, args.epochs+1):
        trl = train(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {ep}/{args.epochs} — Train Loss: {trl:.4f}")
        total_loss = 0
        for src, inp, out, sm, tm in val_loader:
            logits = model(src.to(device), inp.to(device), src_key_padding_mask=sm.to(device), tgt_key_padding_mask=tm.to(device), tgt_mask=generate_square_subsequent_mask(inp.size(0)).to(device))
            total_loss += F.cross_entropy(logits.view(-1, vocab_size), out.to(device).view(-1), ignore_index=0, label_smoothing=0.1).item()
        vl = total_loss / len(val_loader)
        va = eval_token_accuracy(model, val_loader, device)
        print(f" -> Val Loss: {vl:.4f} — Val Acc: {va:.1f}%")
        if vl < best_val - 1e-4:
            best_val, no_imp = vl, 0
            torch.save(model.state_dict(), 'seq2seq_model.pt')
        else:
            no_imp += 1
            if no_imp >= patience:
                print("Early stopping.")
                break
    print("Training complete.")

if __name__=='__main__':
    main()

# inference.py (unchanged)
