import torch, torch.nn.functional as F
from models import MiniRPCTransformer
from tokenizers import Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 128
BOS, EOS, PAD = 1, 2, 0          # adjust if you used other IDs

# ----- load stuff -----
tok = Tokenizer.from_file("rpc_tokenizer.json")
model = MiniRPCTransformer(
    vocab_size = tok.get_vocab_size(with_added_tokens=True),
    d_model    = 256, n_heads = 4,
    n_enc_layers = 4, n_dec_layers = 4,
    max_len = MAX_LEN, pad_id = PAD
).to(DEVICE)
model.load_state_dict(torch.load("runs/XXX_best_model.pt", map_location=DEVICE))
model.eval()

# ----- pick a line straight from the training file -----
line = "click on more info"          # replace with an actual training sentence
ids  = tok.encode(line).ids
src  = torch.tensor([BOS] + ids + [EOS], dtype=torch.long).unsqueeze(0).to(DEVICE)

# ----- autoregressive generation -----
enc_out = None
tgt_ids = [BOS]                 # start with BOS
for _ in range(MAX_LEN):
    tgt_in = torch.tensor(tgt_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(src, tgt_in)          # (1, T, V)
    next_id = int(logits[0, -1].argmax())    # greedy; switch to sampling if you like
    if next_id == EOS: break
    tgt_ids.append(next_id)

decoded = tok.decode(tgt_ids[1:])            # drop BOS
print("INPUT :", line)
print("OUTPUT:", decoded)

