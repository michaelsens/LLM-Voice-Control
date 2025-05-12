
import argparse, math, time, torch, torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from models import MiniRPCTransformer   # earlier code

import json, matplotlib.pyplot as plt, pathlib, time
loss_log = pathlib.Path("loss_history.json")
train_losses, val_losses = [], []

# CLI
p = argparse.ArgumentParser()
p.add_argument("--train", required=True)        
p.add_argument("--val",   required=True)       
p.add_argument("--chkpt", required=True)        
p.add_argument("--epochs", type=int, default=100)
p.add_argument("--batch",  type=int, default=64)
p.add_argument("--lr",     type=float, default=3e-4)
p.add_argument("--warmup", type=int, default=250)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()
device = torch.device(args.device)


# 2. data
train_data = torch.load(args.train)   
val_data   = torch.load(args.val)

def collate(batch, pad_id=0):
    def pad(seqs):
        L = max(len(s) for s in seqs)
        out = torch.full((len(seqs), L), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : len(s)] = s
        return out
    src = pad([b["src"] for b in batch])
    tgt = pad([b["tgt"] for b in batch])
    return src, tgt

train_loader = DataLoader(train_data, batch_size=args.batch,
                          shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(val_data,   batch_size=args.batch,
                          shuffle=False, collate_fn=collate)

PAD_ID = 0   # change if your tokeniser uses a different pad id
def max_id(data):
    m = 0
    for d in data:
        m = max(m, int(d["src"].max()), int(d["tgt"].max()))
    return m

VOCAB = max_id(train_data) + 1
print("Detected vocab size:", VOCAB)

# 3. model
model = MiniRPCTransformer(
    vocab_size=VOCAB,
    d_model=256, n_heads=4,
    n_enc_layers=4, n_dec_layers=4,
    max_len=128, dropout=0.1,
    pad_id=PAD_ID,
).to(device)

optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.LambdaLR(
    optim,
    lambda s: min((s + 1) / args.warmup,
                  (args.warmup ** 0.5) / math.sqrt(s + 1))
)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
best_val  = float("inf")


# 4. train

for epoch in range(1, args.epochs + 1):
    # -------- TRAIN PHASE --------
    model.train()
    train_sum, train_count = 0.0, 0        # NEW
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        logits = model(src, tgt[:, :-1])
        loss = criterion(logits.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step(); optim.zero_grad(); sched.step()

        train_sum += loss.item() * src.size(0)   # NEW
        train_count += src.size(0)               # NEW

    train_loss = train_sum / train_count         # NEW

    # -------- VALIDATION PHASE --------
    model.eval(); val_sum, val_count = 0.0, 0
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            logits = model(src, tgt[:, :-1])
            l = criterion(logits.reshape(-1, VOCAB), tgt[:, 1:].reshape(-1))
            val_sum += l.item() * src.size(0)
            val_count += src.size(0)
    val_loss = val_sum / val_count

    # -------- LOGGING --------
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    with loss_log.open("w") as f:                        # saves every epoch
        json.dump({"train": train_losses,
                   "val":   val_losses}, f)

    print(f"Epoch {epoch:2d} | train {train_loss:.4f} | val {val_loss:.4f}")

    # -------- CHECKPOINT --------
    if val_loss < best_val:
        best_val = val_loss
        Path(args.chkpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(),
                    "vocab": VOCAB,
                    "pad_id": PAD_ID}, args.chkpt)
        print("   ✓ checkpoint updated")

print("Done. Best val loss:", best_val)

# ---------- OPTIONAL PLOT AFTER TRAINING ----------
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="train")
plt.plot(val_losses,  label="valid")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.yscale("log")
plt.title("Training vs Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curve.png", dpi=200)   # opens in any image viewer
print("Plot saved → loss_curve.png")