# sample command to run this file: python train.py --data ../expanded.jsonl --block_size 128  --batch 16 --epochs 20 --lr 3e-4 --val_ratio 0.1
# after training you should get a seq2seq_model.pt
# execute "py inference.py" and it will use that model so you can test on it

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
from tqdm import tqdm

# Enable faster cuDNN kernels
torch.backends.cudnn.benchmark = True

# ----------------------------------------------------------------------------
# Local Attention Mask
# ----------------------------------------------------------------------------
def generate_local_attention_mask(seq_len, window_size):
    mask = torch.full((seq_len, seq_len), float('-inf'))
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i+1] = 0
    return mask

# ----------------------------------------------------------------------------
# Positional Encoding
# ----------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(1))  # [max_len,1,d_model]

    def forward(self, x):  # x: [seq_len, batch, d_model]
        return x + self.pe[:x.size(0)]

# ----------------------------------------------------------------------------
# Seq2Seq Transformer Model
# ----------------------------------------------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3,
                 dim_feedforward=2048, dropout=0.1,
                 max_len=130, window_size=32):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.tgt_emb = nn.Embedding(vocab_size, d_model)
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
        src_emb = self.pos_enc(self.src_emb(src) * (self.d_model ** 0.5))
        tgt_emb = self.pos_enc(self.tgt_emb(tgt) * (self.d_model ** 0.5))
        tgt_mask = generate_local_attention_mask(tgt.size(0), self.window_size).to(src.device)
        memory = self.transformer.encoder(src_emb)
        outs = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.generator(outs)

# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, examples, enc, block_size):
        BOS_ID = enc.n_vocab
        EOS_ID = enc.n_vocab + 1
        self.examples = []
        for utt, rpc_str in examples:
            src_ids = enc.encode(utt)[:block_size]
            tgt_ids = enc.encode(rpc_str)[: block_size - 1]
            if not src_ids or not tgt_ids:
                continue
            inp_ids = [BOS_ID] + tgt_ids
            out_ids = tgt_ids + [EOS_ID]
            self.examples.append((
                torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(inp_ids, dtype=torch.long),
                torch.tensor(out_ids, dtype=torch.long)
            ))
        if not self.examples:
            raise ValueError("No valid examples found in dataset.")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# ----------------------------------------------------------------------------
# Collate function
# ----------------------------------------------------------------------------
def collate_fn(batch):
    src, inp, out = zip(*batch)
    max_src = max(s.size(0) for s in src)
    max_tgt = max(t.size(0) for t in inp)
    bs = len(src)
    src_pad = torch.zeros(max_src, bs, dtype=torch.long)
    inp_pad = torch.zeros(max_tgt, bs, dtype=torch.long)
    out_pad = torch.zeros(max_tgt, bs, dtype=torch.long)
    for i, (s, inp_seq, out_seq) in enumerate(batch):
        src_pad[:s.size(0), i]       = s
        inp_pad[:inp_seq.size(0), i] = inp_seq
        out_pad[:out_seq.size(0), i] = out_seq
    return src_pad, inp_pad, out_pad

# ----------------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------------
def train(model, loader, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = 0.0
    for src, inp, out in tqdm(loader, desc="Training"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(src, inp)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                out.view(-1), ignore_index=0
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ----------------------------------------------------------------------------
# Fast token-level evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_token_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for src, inp, out in tqdm(loader, desc="Evaluating"):
        src, inp, out = src.to(device), inp.to(device), out.to(device)
        logits = model(src, inp)
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
    parser.add_argument('--data',      required=True,
                        help="Path to JSONL file with utterance/rpc pairs")
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--d_model',     type=int, default=512)
    parser.add_argument('--nhead',       type=int, default=8)
    parser.add_argument('--enc_layers',  type=int, default=3)
    parser.add_argument('--dec_layers',  type=int, default=3)
    parser.add_argument('--ff',          type=int, default=2048)
    parser.add_argument('--dropout',     type=float, default=0.1)
    parser.add_argument('--batch',       type=int, default=16)
    parser.add_argument('--epochs',      type=int, default=20)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--device',      default='cuda:0')
    parser.add_argument('--val_ratio',   type=float, default=0.1,
                        help="Fraction of data to hold out for validation")
    args = parser.parse_args()

    # Load and prepare examples
    examples = []
    with open(args.data, 'r', encoding='utf-8') as f:
        for line in f:
            j = json.loads(line)
            utt = j['utterance'].strip()
            rpc_str = json.dumps(j['rpc'], separators=(',', ':'), ensure_ascii=False)
            examples.append((utt, rpc_str))

    enc = tiktoken.get_encoding('gpt2')
    dataset = Seq2SeqDataset(examples, enc, args.block_size)

    # Split dataset
    total = len(dataset)
    val_size = int(args.val_ratio * total)
    train_size = total - val_size
    print(f"Dataset size: {total}, train: {train_size}, val: {val_size}")
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              collate_fn=collate_fn, num_workers=4)

    # Model setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    vocab_size = enc.n_vocab + 2  # account for BOS + EOS
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        max_len=args.block_size,
        window_size=args.block_size // 4
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs,
        pct_start=0.1
    )
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, train_loader, optimizer, scheduler, scaler, device)
        print(f"Epoch {epoch}/{args.epochs} — Train Loss: {train_loss:.4f}")

        # Validation loss
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for src, inp, out in tqdm(val_loader, desc="Validating"):
                src, inp, out = src.to(device), inp.to(device), out.to(device)
                logits = model(src, inp)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    out.view(-1), ignore_index=0
                )
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Fast token accuracy
        val_acc = eval_token_accuracy(model, val_loader, device)
        print(f" -> Val Loss: {val_loss:.4f} — Val Token-Acc: {val_acc:.1f}%")

    # Save model
    torch.save(model.state_dict(), 'seq2seq_model.pt')
    print("Model saved to seq2seq_model.pt")

if __name__ == '__main__':
    main()
