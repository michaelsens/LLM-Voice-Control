import argparse
import json
import re
import torch
import tiktoken
from train import Seq2SeqTransformer

# A small helper to clean & normalize domains
def normalize_url(domain: str) -> str:
    # strip trailing punctuation
    domain = domain.strip().rstrip('.,!?')
    # if no scheme, add https
    if not re.match(r'https?://', domain):
        domain = 'https://' + domain
    return domain

def infer(model, enc, device, utterance, block_size):
    # 1) Tokenize
    src_ids = enc.encode(utterance)[:block_size]
    src = torch.tensor(src_ids, dtype=torch.long).unsqueeze(1).to(device)

    BOS, EOS = enc.n_vocab, enc.n_vocab + 1
    ys = torch.tensor([BOS], dtype=torch.long).unsqueeze(1).to(device)

    # 2) Greedy generate
    for _ in range(block_size):
        logits = model(src, ys)           # [tgt_len,1,vocab]
        next_id = logits[-1,0].argmax().unsqueeze(0).unsqueeze(1)
        ys = torch.cat([ys, next_id], dim=0)
        if next_id.item() == EOS:
            break

    # 3) Decode raw JSON
    token_ids = ys.squeeze().tolist()[1:-1]
    raw = enc.decode(token_ids)
    try:
        rpc = json.loads(raw)
    except json.JSONDecodeError:
        # simple repair
        r = raw.strip().rstrip(', ')
        if not r.endswith('}'): r += '}'
        rpc = json.loads(r)

    # 4) If it's a navigate intent, override the URL by regex
    if rpc.get("method") == "navigate":
        m = re.search(r'\b(?:go to|open|navigate to|load)\s+(\S+)', utterance, flags=re.IGNORECASE)
        if m:
            domain = m.group(1)
            rpc["params"]["url"] = normalize_url(domain)

    return rpc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='seq2seq_model.pt')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--device',    default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    enc    = tiktoken.get_encoding('gpt2')
    vocab  = enc.n_vocab + 2

    # rebuild & load
    model = Seq2SeqTransformer(vocab)
    ckpt  = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    print(f"Loaded model from {args.model_path} on {device}")

    while True:
        utt = input("Enter utterance> ").strip()
        if not utt:
            continue
        try:
            rpc = infer(model, enc, device, utt, args.block_size)
            print("Predicted RPC:", json.dumps(rpc, ensure_ascii=False))
        except Exception as e:
            print("Error during inference:", e)

if __name__ == '__main__':
    main()
