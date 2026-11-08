#!/usr/bin/env python3
"""Sample a checkpoint several times per prompt, so checkpoint comparisons are
not decided by one lucky draw at temperature 0.7.

Usage:
    python3 eval_chat.py --ckpt blacksmith-1_sft.pt --samples 5
    python3 eval_chat.py --ckpt x.pt --samples 5 --prompts prompts.txt

Each prompt starts a fresh conversation, formatted exactly like generate.py:
    <|endoftext|><|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n
"""
import argparse
import torch
import torch.nn.functional as F

from model import NanoGPT
from tokenmaker import Tokenizer, TokenizerNoIM, TokenizerVocabSize
from hyperparameters import d_model, n_heads, n_layers, max_seq_len

DEFAULT_PROMPTS = [
    'Hello! Who are you?',
    'What is the capital of France?',
    'What is 2+2?',
    'Explain why the sky is blue in one paragraph.',
    'Summarize the plot of Hamlet in three sentences.',
    'Write a Python function that reverses a string.',
    'Tell me a short story about a dragon.',
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--max-new', type=int, default=200)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-k', type=int, default=25)
    p.add_argument('--repeat-penalty', type=float, default=1.3)
    p.add_argument('--prompts', default=None, help='file with one prompt per line')
    p.add_argument('--no-eot-between-turns', action='store_true',
                   help='omit the leading EOT (matches generate.py --no-eot-between-turns)')
    return p.parse_args()


def main():
    args = parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = NanoGPT(TokenizerVocabSize, d_model, n_heads, n_layers, max_seq_len).to(dev)
    model = model.to(torch.bfloat16)
    chk = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state = {k.replace('module.', '', 1): v for k, v in chk.get('model', chk).items()}
    model.load_state_dict(state)
    model.eval()
    print(f'loaded {args.ckpt} (step {chk.get("step", "?")})', flush=True)

    EOT = Tokenizer.eos_token_id
    IM_START = Tokenizer.encode('<|im_start|>')[0]
    IM_END = Tokenizer.encode('<|im_end|>')[0]
    NL = Tokenizer.encode('\n')[0]
    ROLE_USER = Tokenizer.encode('user')
    ROLE_ASSIST = Tokenizer.encode('assistant')

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        prompts = [ln.strip() for ln in open(args.prompts) if ln.strip()]

    @torch.no_grad()
    def sample(prompt, seed):
        torch.manual_seed(seed)
        ids = [] if args.no_eot_between_turns else [EOT]
        ids += [IM_START] + ROLE_USER + [NL]
        ids += TokenizerNoIM.encode(prompt)
        ids += [IM_END, NL, IM_START] + ROLE_ASSIST + [NL]

        x = torch.tensor([ids], device=dev)
        logits, k, v = model(x, infer=True)
        pos = x.shape[1]
        recent = torch.tensor([], dtype=torch.long, device=dev)
        out = []
        for _ in range(args.max_new):
            lg = logits[0, -1, :] / args.temperature
            if args.repeat_penalty != 1.0 and recent.numel():
                g = lg.gather(0, recent)
                lg.scatter_(0, recent, torch.where(g < 0, g * args.repeat_penalty,
                                                   g / args.repeat_penalty))
            if args.top_k > 0:
                thr = torch.topk(lg, args.top_k).values[-1]
                lg = torch.where(lg < thr, torch.tensor(float('-inf'), device=dev), lg)
            tok = torch.multinomial(F.softmax(lg, -1), 1)
            if tok.item() in (EOT, IM_END):
                break
            out.append(tok.item())
            recent = torch.cat([recent, tok])
            if recent.numel() > 64:
                recent = recent[-64:]
            logits, k, v = model(tok[None, :], k, v, infer=True, pos_offset=pos)
            pos += 1
        return Tokenizer.decode(out)

    for i, prompt in enumerate(prompts):
        print(f'\n=== {prompt}')
        for s in range(args.samples):
            print(f'  [{s}] {sample(prompt, seed=1000 * (i + 1) + s)!r}', flush=True)


if __name__ == '__main__':
    main()
