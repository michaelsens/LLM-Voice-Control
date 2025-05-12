# train.py
# Usage:
# python train.py --data ../expanded.jsonl --block_size 128 --batch 16 --epochs 40 --lr 3e-4 --val_ratio 0.1 --device cuda:0

import argparse
import json
import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Rotary positional embeddings
# ----------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_freq=10.0):
        super().__init__()
        inv_freq = 1.0 / (max_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        return torch.cat((freqs.sin(), freqs.cos()), dim=-1)

# ----------------------------------------------------------------------------
# Dataset: raw utterance → JSON RPC
# ----------------------------------------------------------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, examples, enc, block_size):
        BOS, EOS = enc.n_vocab, enc.n_vocab + 1
        self.data = []
        for utt, rpc in examples:
            src_ids = enc.encode(utt)[:block_size]
            tgt_str = json.dumps(rpc, separators=(',',':'))
            tgt_ids = enc.encode(tgt_str)[:block_size-1]
            if not tgt_ids: continue
            self.data.append((
                torch.tensor([BOS] + src_ids, dtype=torch.long),
                torch.tensor([BOS] + tgt_ids, dtype=torch.long),
                torch.tensor(tgt_ids + [EOS], dtype=torch.long)
            ))
        if not self.data:
            raise ValueError("No valid examples found.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

# ----------------------------------------------------------------------------
# Collate with padding
# ----------------------------------------------------------------------------
def collate_fn(batch):
    src, inp, out = zip(*batch)
    max_s = max(map(len, src)); max_t = max(map(len, inp))
    B = len(src)
    src_pad = torch.zeros(max_s, B, dtype=torch.long)
    inp_pad = torch.zeros(max_t, B, dtype=torch.long)
    out_pad = torch.zeros(max_t, B, dtype=torch.long)
    for i, (s, ii, oo) in enumerate(batch):
        src_pad[:len(s), i] = s
        inp_pad[:len(ii), i] = ii
        out_pad[:len(oo), i] = oo
    src_mask = (src_pad == 0).transpose(0,1)
    tgt_mask = (inp_pad == 0).transpose(0,1)
    return src_pad, inp_pad, out_pad, src_mask, tgt_mask

# ----------------------------------------------------------------------------
# True cross-attention seq2seq
# ----------------------------------------------------------------------------
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_enc=3, num_dec=3, ff=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.shared_emb = nn.Embedding(vocab_size, d_model)
        self.pos = RotaryEmbedding(d_model)
        enc_layer = TransformerEncoderLayer(d_model,nhead,ff,dropout)
        self.encoder = TransformerEncoder(enc_layer, num_enc)
        dec_layer = TransformerDecoderLayer(d_model,nhead,ff,dropout)
        self.decoder = TransformerDecoder(dec_layer, num_dec)
        self.generator = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt, src_key_padding_mask=None,
                tgt_key_padding_mask=None, tgt_mask=None):
        # embed + scale
        src_e = self.shared_emb(src) * math.sqrt(self.d_model)
        tgt_e = self.shared_emb(tgt) * math.sqrt(self.d_model)
        # rotary
        Ls, Lt = src_e.size(0), tgt_e.size(0)
        src_e = src_e + self.pos(Ls, src.device).unsqueeze(1)
        tgt_e = tgt_e + self.pos(Lt, tgt.device).unsqueeze(1)
        # encode & decode
        memory = self.encoder(src_e, src_key_padding_mask=src_key_padding_mask)
        out = self.decoder(tgt_e, memory,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=src_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
        return self.generator(out)

# ----------------------------------------------------------------------------
# Causal mask
# ----------------------------------------------------------------------------
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz,sz), float('-inf')), diagonal=1)

# ----------------------------------------------------------------------------
# Run one epoch (train or eval)
# ----------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, scheduler, device, is_train):
    total_loss = 0.0
    total_tokens = 0
    if is_train:
        model.train()
    else:
        model.eval()
    mask_cache = {}
    with torch.set_grad_enabled(is_train):
        for src, inp, out, sm, tm in tqdm(loader, desc="Train" if is_train else "Valid"):
            src, inp, out = src.to(device), inp.to(device), out.to(device)
            sm, tm = sm.to(device), tm.to(device)
            T = inp.size(0)
            # causal mask (cached by length)
            if T not in mask_cache:
                mask_cache[T] = generate_square_subsequent_mask(T).to(device)
            tgt_mask = mask_cache[T]
            logits = model(src, inp,
                           src_key_padding_mask=sm,
                           tgt_key_padding_mask=tm,
                           tgt_mask=tgt_mask)
            # per-token loss
            loss_all = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                out.view(-1),
                reduction='none', ignore_index=0, label_smoothing=0.1
            )
            mask = (out.view(-1) != 0).float()
            loss = (loss_all * mask).sum() / mask.sum()
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            tokens = mask.sum().item()
            total_loss += loss.item() * tokens
            total_tokens += tokens
    # return average per-token loss
    return total_loss / total_tokens

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      required=True)
    parser.add_argument('--block_size',type=int, default=128)
    parser.add_argument('--batch',     type=int, default=16)
    parser.add_argument('--epochs',    type=int, default=40)
    parser.add_argument('--lr',        type=float, default=3e-4)
    parser.add_argument('--device',    default='cpu')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # load examples
    exs = []
    with open(args.data,'r',encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            exs.append((j['utterance'], j['rpc']))

    enc = tiktoken.get_encoding('gpt2')
    dataset = Seq2SeqDataset(exs, enc, args.block_size)
    N = len(dataset)
    val_n = int(args.val_ratio * N)
    train_n = N - val_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42))
    lw = dict(collate_fn=collate_fn, pin_memory=True,
              batch_size=args.batch, num_workers=min(4, os.cpu_count() or 1))
    train_loader = DataLoader(train_ds, shuffle=True,  **lw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **lw)

    # model, optimizer, scheduler
    vocab = enc.n_vocab + 2
    model = Seq2SeqTransformer(vocab).to(device)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1)

    train_losses, val_losses = [], []
    for ep in range(1, args.epochs+1):
        tl = run_epoch(model, train_loader, opt, sched, device, True)
        vl = run_epoch(model, val_loader,   opt, sched, device, False)
        train_losses.append(tl)
        val_losses.append(vl)
        print(f"Epoch {ep}/{args.epochs} — Train Loss: {tl:.4f}  Val Loss: {vl:.4f}")

    # plot & save
    plt.figure()
    xs = list(range(1, args.epochs+1))
    plt.plot(xs, train_losses, label='Train')
    plt.plot(xs, val_losses,   label='Valid')
    plt.xlabel('Epoch'); plt.ylabel('Per-Token CE Loss')
    plt.title('Training & Validation')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()
    print("Saved loss curve to loss_curve.png")

    # final save
    torch.save(model.state_dict(), 'seq2seq_model.pt')
    print("Training complete. Model saved to seq2seq_model.pt")

if __name__ == '__main__':
    main()