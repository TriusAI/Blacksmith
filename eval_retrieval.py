#!/usr/bin/env python3
"""Needle-in-a-haystack test for the Blacksmith checkpoints.

Places a passphrase at a given depth inside a long filler context, asks for it,
and checks whether the answer contains it. Sweeping length vs depth gives a
capability curve rather than a single anecdote, which is the only way to say
whether the 64k training and the YaRN extension to 128k actually work.

    python3 eval_retrieval.py --ckpt blacksmith-1_sft.pt \
        --lengths 4096 16384 32768 65536 --depths 0.0 0.5 1.0

For lengths beyond the trained 65536, pass the YaRN parameters that match the
converter's metadata:

    ... --lengths 131072 --rope-scale 2 --original-max-pos 65536
"""
import argparse
import json

import torch
import torch.nn.functional as F

from model import NanoGPT
from tokenmaker import Tokenizer, TokenizerVocabSize
from hyperparameters import d_model, n_heads, n_layers

PASSPHRASE = 'BLUE-HERRING-42'
NEEDLE = (f'\n\nThe secret passphrase is {PASSPHRASE}.\n\n')
QUESTION = 'What is the secret passphrase mentioned above? Answer with the passphrase only.'
# A single neutral paragraph repeated to the requested length. Using one fixed
# text keeps the local context identical at every length, so a failure can be
# attributed to distance/position rather than to whatever document happened to
# end at that token offset.
FILLER_TEXT = (
    'The museum opened to the public in the spring of that year, and its first '
    'exhibition was a survey of nineteenth century landscape painting. Visitors '
    'walked through rooms arranged by region rather than by date, which was '
    'unusual at the time and drew comment from several newspapers. The building '
    'itself had been a warehouse, and the architects kept the iron columns and '
    'the wide timber floors, adding only skylights and a new staircase. In the '
    'decades that followed the collection grew slowly, mostly through gifts, and '
    'the museum became known for the quietness of its upper galleries. '
)


def build_filler(n_tokens):
    """One paragraph repeated to the requested length."""
    unit = Tokenizer.encode(FILLER_TEXT)
    reps = n_tokens // len(unit) + 1
    return (unit * reps)[:n_tokens]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--lengths', type=int, nargs='+', default=[4096, 16384, 32768, 65536])
    p.add_argument('--depths', type=float, nargs='+', default=[0.0, 0.5, 1.0],
                   help='0 = needle at the very start, 1 = immediately before the question')
    p.add_argument('--rope-scale', type=float, default=1.0,
                   help='YaRN stretch factor for lengths beyond the trained context')
    p.add_argument('--original-max-pos', type=int, default=65536,
                   help='context the base was trained at; YaRN ramp reference')
    p.add_argument('--max-new', type=int, default=24)
    p.add_argument('--samples', type=int, default=1,
                   help='generations per (length, depth); >1 exposes variance')
    p.add_argument('--temperature', type=float, default=0.0,
                   help='0 = greedy, the usual setting for retrieval tests')
    p.add_argument('--mode', choices=['chat', 'completion'], default='chat',
                   help='chat asks a question in the SFT format; completion makes '
                        'the model repeat the sentence, which tests copying '
                        'without requiring instruction following')
    return p.parse_args()


def main():
    args = parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_len = max(args.lengths)

    model = NanoGPT(TokenizerVocabSize, d_model, n_heads, n_layers, max_len,
                    rope_scale=args.rope_scale,
                    original_max_pos=args.original_max_pos).to(dev)
    model = model.to(torch.bfloat16)
    chk = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state = {k.replace('module.', '', 1): v for k, v in chk.get('model', chk).items()}
    model.load_state_dict(state)
    model.eval()
    print(f'loaded {args.ckpt} (step {chk.get("step", "?")}), '
          f'rope_scale={args.rope_scale}, orig_ctx={args.original_max_pos}', flush=True)

    EOT = Tokenizer.eos_token_id
    IM_START = Tokenizer.encode('<|im_start|>')[0]
    IM_END = Tokenizer.encode('<|im_end|>')[0]
    NL = Tokenizer.encode('\n')[0]
    ROLE_USER = Tokenizer.encode('user')
    ROLE_ASSIST = Tokenizer.encode('assistant')

    filler = build_filler(max_len + 4096)
    needle_ids = Tokenizer.encode(NEEDLE)
    question_ids = Tokenizer.encode('\n\n' + QUESTION)

    @torch.no_grad()
    def run(length, depth):
        # Reserve room for the needle, the question and the reply.
        body_len = length - len(needle_ids) - len(question_ids)
        if body_len < 64:
            return None, ''
        body = filler[:body_len]
        cut = int(round(depth * body_len))
        body = body[:cut] + needle_ids + body[cut:]

        if args.mode == 'completion':
            # No chat wrapper: continue the sentence the needle started. This is
            # the fairest probe for a base model, which never saw a long
            # document followed by an instruction inside one turn.
            tail = Tokenizer.encode('\n\nReminder: the secret passphrase is')
            ids = list(body) + tail
        else:
            ids = [EOT, IM_START] + ROLE_USER + [NL] + body + question_ids
            ids += [IM_END, NL, IM_START] + ROLE_ASSIST + [NL]
        if len(ids) > length:
            ids = ids[:length]

        x = torch.tensor([ids], device=dev)
        logits, k, v = model(x, infer=True)
        pos = x.shape[1]
        out = []
        for _ in range(args.max_new):
            lg = logits[0, -1, :]
            tok = (lg.argmax() if args.temperature == 0
                   else torch.multinomial(F.softmax(lg / args.temperature, -1), 1)[0])
            if tok.item() in (EOT, IM_END):
                break
            out.append(tok.item())
            logits, k, v = model(tok[None, None], k, v, infer=True, pos_offset=pos)
            pos += 1
        text = Tokenizer.decode(out)
        return PASSPHRASE in text, text

    print(f'\n{"length":>8} {"depth":>6}  hit  reply')
    table = {}
    for length in args.lengths:
        for depth in args.depths:
            results = []
            for _ in range(args.samples):
                try:
                    hit, text = run(length, depth)
                except RuntimeError as e:
                    print(f'{length:>8} {depth:>6.2f}  ERR  {str(e)[:60]}')
                    break
                if hit is None:
                    break
                results.append(hit)
                print(f'{length:>8} {depth:>6.2f}  {"Y" if hit else "n":>3}  {text[:70]!r}', flush=True)
            if results:
                table[(length, depth)] = sum(results) / len(results)

    if table:
        print('\nsummary')
        for length in args.lengths:
            cells = [table.get((length, d)) for d in args.depths if (length, d) in table]
            if cells:
                print(f'  {length:>7}: {sum(cells) / len(cells):.2f} avg hit rate')


if __name__ == '__main__':
    main()
