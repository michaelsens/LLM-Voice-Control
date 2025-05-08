import argparse
import time
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm

# Enable faster cuDNN kernels
torch.backends.cudnn.benchmark = True

def generate_square_subsequent_mask(sz):
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# ============================================================================
# Positional Encoding
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))  # [max_len,1,d_model]

    def forward(self, x):  # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0)]

# ============================================================================
# Seq2Seq Transformer Model
# ============================================================================
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, max_len=130):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu'
        )
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        src_emb = self.pos_enc(self.src_tok_emb(src) * (self.d_model ** 0.5))
        tgt_emb = self.pos_enc(self.tgt_tok_emb(tgt) * (self.d_model ** 0.5))
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        memory = self.transformer.encoder(src_emb)
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.generator(outs)

# ============================================================================
# Dataset for Seq2Seq
# ============================================================================
class Seq2SeqDataset(Dataset):
    def __init__(self, path, enc, block_size):
        BOS_ID = enc.n_vocab
        EOS_ID = enc.n_vocab + 1
        self.examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                utt = data['utterance'].strip()
                rpc = json.dumps(data['rpc'], ensure_ascii=False)
                src_ids = enc.encode(utt)[:block_size]
                tgt_ids = enc.encode(rpc)[:block_size]
                if not src_ids or not tgt_ids:
                    continue
                inp = [BOS_ID] + tgt_ids
                out = tgt_ids + [EOS_ID]
                self.examples.append((
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(inp, dtype=torch.long),
                    torch.tensor(out, dtype=torch.long)
                ))
        if not self.examples:
            raise ValueError("No valid examples.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch):
    src_seqs, inp_seqs, out_seqs = zip(*batch)
    max_src = max(s.size(0) for s in src_seqs)
    max_tgt = max(t.size(0) for t in inp_seqs)
    vs = len(src_seqs)
    src_pad = torch.zeros(max_src, vs, dtype=torch.long)
    inp_pad = torch.zeros(max_tgt, vs, dtype=torch.long)
    out_pad = torch.zeros(max_tgt, vs, dtype=torch.long)
    for i, (s, inp, out) in enumerate(batch):
        src_pad[:s.size(0), i] = s
        inp_pad[:inp.size(0), i] = inp
        out_pad[:out.size(0), i] = out
    return src_pad, inp_pad, out_pad

# ============================================================================
# Training & Inference
# ============================================================================

def train(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for src, inp, out in tqdm(loader, desc="Training"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(src, inp)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), out.view(-1), ignore_index=0)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def infer(model, enc, device, utterance, sep_token, max_len):
    model.eval()
    src_ids = enc.encode(utterance)
    src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(1)
    BOS_ID = enc.n_vocab
    EOS_ID = enc.n_vocab + 1
    ys = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
    for _ in range(max_len):
        logits = model(src, ys)
        next_id = torch.argmax(logits[-1, 0]).item()
        ys = torch.cat([ys, torch.tensor([[next_id]], device=device)], dim=0)
        if next_id == EOS_ID:
            break
    token_ids = ys[1:-1].squeeze(1).cpu().tolist()
    return enc.decode(token_ids)

@torch.no_grad()
def evaluate(model, loader, enc, device, sep_token, max_len):
    model.eval()
    correct = total = 0
    for src, inp, out in tqdm(loader, desc="Evaluating"):
        utt = enc.decode(src[:, 0].tolist())
        pred = infer(model, enc, device, utt, sep_token, max_len)
        gold = enc.decode(out[:, 0].cpu().tolist()).strip()
        if pred.strip() == gold:
            correct += 1
        total += 1
    return 100 * correct / total if total else 0.0

# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--test_data')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--enc_layers', type=int, default=3)
    parser.add_argument('--dec_layers', type=int, default=3)
    parser.add_argument('--ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()

    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab + 2
    sep_token = "<SEP>"

    ds = Seq2SeqDataset(args.data, enc, args.block_size)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        collate_fn=collate_fn, num_workers=4)

    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        max_len=args.block_size + 2
    )
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(loader), epochs=args.epochs, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs+1):
        train_loss = train(model, loader, optimizer, scheduler, scaler, device)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {train_loss:.4f}")
        if args.test_data:
            test_ds = Seq2SeqDataset(args.test_data, enc, args.block_size)
            test_loader = DataLoader(test_ds, batch_size=1, collate_fn=collate_fn)
            val_acc = evaluate(model, test_loader, enc, device, sep_token, args.block_size)
            print(f"Validation Acc: {val_acc:.1f}%")

    if args.infer:
        while True:
            utt = input("Enter utterance> ")
            print("Predicted RPC:", infer(model, enc, device, utt, sep_token, args.block_size))

if __name__ == '__main__':
    main()
