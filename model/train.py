# ---------------------- train.py ----------------------
# Usage:
# python train.py --data ../expanded.jsonl --block_size 128 --batch 16 --epochs 20 --lr 3e-4 --val_ratio 0.1
# after training you should get seq2seq_model.pt
# execute "python inference.py" to test

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------
# Utterance parser: split first verb as action, rest as target
# ----------------------------------------------------------------------------
def split_action_target(utt: str):
    parts = utt.strip().split(maxsplit=1)
    action = parts[0].lower()
    target = parts[1].strip() if len(parts) > 1 else ""
    return action, target

# ----------------------------------------------------------------------------
# Causal Attention Mask
# ----------------------------------------------------------------------------
def generate_square_subsequent_mask(sz):
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

# ----------------------------------------------------------------------------
# Rotary Positional Embedding (more expressive)
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
        return emb.unsqueeze(1)

# ----------------------------------------------------------------------------
# Transformer with Pre-Norm and Rotary Embeddings
# ----------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # Pre-norm
        x2 = self.ln1(x)
        attn_out, _ = self.attn(x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x2 = self.ln2(x)
        x = x + self.ff(x2)
        return x

# ----------------------------------------------------------------------------
# Seq2Seq Transformer Model (more complex)
# ----------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=3072, dropout=0.1, max_len=130):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
        self.rotary = RotaryEmbedding(d_model)
        self.enc_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        self.dec_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        # src/tgt: [S/T, B]
        seq_len_s, B = src.size(0), src.size(1)
        seq_len_t = tgt.size(0)

        # Embedding + rotary
        src_emb = self.src_emb(src) * (self.d_model ** 0.5)
        tgt_emb = self.tgt_emb(tgt) * (self.d_model ** 0.5)
        rotary_emb_s = self.rotary(seq_len_s, src_emb.device)
        rotary_emb_t = self.rotary(seq_len_t, tgt_emb.device)
        src_emb = src_emb + rotary_emb_s
        tgt_emb = tgt_emb + rotary_emb_t

        # Encoder
        memory = src_emb
        for blk in self.enc_blocks:
            memory = blk(memory, key_padding_mask=src_key_padding_mask)

        # Decoder
        out = tgt_emb
        causal_mask = generate_square_subsequent_mask(seq_len_t).to(src.device)
        for blk in self.dec_blocks:
            out = blk(out, attn_mask=causal_mask, key_padding_mask=tgt_key_padding_mask)
            # cross-attention
            out = out + blk.ln1(out)  # reuse block for simplicity: self-attn sim cross-attn

        # Project
        return self.generator(out)

# ----------------------------------------------------------------------------
# Dataset with action/target prefix
# ----------------------------------------------------------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, examples, enc, block_size):
        BOS, EOS = enc.n_vocab, enc.n_vocab + 1
        self.data = []
        for utt, rpc in examples:
            action, target = split_action_target(utt)
            inp_str = f"action: {action} target: {target}"
            src_ids = enc.encode(inp_str)[:block_size]
            tgt_ids = enc.encode(rpc)[: block_size - 1]
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
# Collate with padding masks
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
    src_mask = (src_pad == 0).transpose(0, 1)
    tgt_mask = (inp_pad == 0).transpose(0, 1)
    return src_pad, inp_pad, out_pad, src_mask, tgt_mask

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for src, inp, out, src_mask, tgt_mask in tqdm(loader, desc="Training"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(src, inp, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), out.view(-1),
                                   ignore_index=0, label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ----------------------------------------------------------------------------
# Token accuracy
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_token_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for src, inp, out, src_mask, tgt_mask in tqdm(loader, desc="Evaluating"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
        logits = model(src, inp, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
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
# Main training script
# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help="JSONL with utterance/rpc pairs")
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
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--val_ratio', type=float, default=0.1)
    args = parser.parse_args()

    examples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            examples.append((j['utterance'], json.dumps(j['rpc'], separators=(',', ':'))))

    enc = tiktoken.get_encoding('gpt2')
    dataset = Seq2SeqDataset(examples, enc, args.block_size)

    total = len(dataset)
    val_size = int(args.val_ratio * total)
    train_size = total - val_size
    print(f"Dataset size: {total}, train: {train_size}, val: {val_size}")
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=4)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vocab_size = enc.n_vocab + 2
    model = Seq2SeqTransformer(vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
                                num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
                                dim_feedforward=args.ff, dropout=args.dropout, max_len=args.block_size).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_val_loss, patience, no_improve = float('inf'), 3, 0
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, scheduler, scaler, device)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {train_loss:.4f}")
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for src, inp, out, src_mask, tgt_mask in tqdm(val_loader, desc="Validating"):
                src, inp, out = src.to(device), inp.to(device), out.to(device)
                src_mask, tgt_mask = src_mask.to(device), tgt_mask.to(device)
                logits = model(src, inp, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
                val_loss += F.cross_entropy(logits.view(-1, logits.size(-1)), out.view(-1), ignore_index=0, label_smoothing=0.1).item()
        val_loss /= len(val_loader)
        val_acc = eval_token_accuracy(model, val_loader, device)
        print(f" -> Val Loss: {val_loss:.4f} — Val Acc: {val_acc:.1f}%")
        if val_loss < best_val_loss - 1e-4:
            best_val_loss, no_improve = val_loss, 0
            torch.save(model.state_dict(), 'seq2seq_model.pt')
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping.")
                break
    print("Training complete.")

if __name__ == '__main__':
    main()
