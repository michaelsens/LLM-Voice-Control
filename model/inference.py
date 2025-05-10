import argparse
import torch
import tiktoken
from train import Seq2SeqTransformer, infer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='seq2seq_model.pt', help='Path to saved model weights')
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # Setup device and tokenizer
    device = torch.device(args.device)
    enc = tiktoken.get_encoding('gpt2')
    vocab_size = enc.n_vocab + 2

    # Reconstruct model architecture and load weights
    model = Seq2SeqTransformer(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=2048,
        dropout=0.1,
        max_len=args.block_size,               # ← same as training
        window_size=args.block_size // 4       # ← same as training
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device).eval()

    print(f"Loaded model from {args.model_path} on {device}")

    # Interactive inference loop
    while True:
        utterance = input("Enter utterance> ")
        if not utterance.strip():
            continue
        rpc = infer(model, enc, device, utterance, args.block_size)
        print("Predicted RPC:", rpc)


if __name__ == '__main__':
    main()
