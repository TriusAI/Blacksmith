#!/usr/bin/env python3
"""Convert LongAlign-10k and HotpotQA into the {"data": [user, assistant, ...]}
JSONL that sft_dataset.py reads.

    python3 convert_datasets.py --longalign LongAlign-10k/long.jsonl \
        --out-longalign longalign_sft.jsonl

    python3 convert_datasets.py --hotpotqa hotpotqa/distractor/train-00000-of-00002.parquet \
        --out-hotpotqa hotpotqa_sft.jsonl --max-examples 20000

LongAlign rows are already chat-formatted, so the conversion is a flattening of
`messages` into an alternating user/assistant list (SFTDataset assigns roles by
position parity, so the order is what matters).

HotpotQA is turned into a retrieval-shaped example: the ten paragraphs of the
distractor setting become the user turn, the question is appended, and the
answer is the assistant turn. Supporting-fact passages stay in the context, but
the paragraph order is shuffled so the answer is not always first.
"""
import argparse
import json
import random

import pyarrow.parquet as pq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--longalign', default=None)
    p.add_argument('--out-longalign', default='longalign_sft.jsonl')
    p.add_argument('--hotpotqa', default=None)
    p.add_argument('--out-hotpotqa', default='hotpotqa_sft.jsonl')
    p.add_argument('--max-examples', type=int, default=20000)
    p.add_argument('--docs', type=int, default=10,
                   help='how many paragraphs to keep per HotpotQA example')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def convert_longalign(src, dst):
    n = kept = 0
    with open(src) as f, open(dst, 'w') as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            row = json.loads(line)
            msgs = row.get('messages') or []
            data = [m.get('content', '') for m in msgs if m.get('role') in ('user', 'assistant')]
            if len(data) < 2 or len(data) % 2 != 0:
                continue        # need a complete user/assistant pairing
            out.write(json.dumps({'data': data}, ensure_ascii=False) + '\n')
            kept += 1
    print(f'longalign: {kept}/{n} examples -> {dst}')


def convert_hotpotqa(src, dst, limit, n_docs, seed):
    rng = random.Random(seed)
    table = pq.read_table(src)
    kept = 0
    with open(dst, 'w') as out:
        for row in table.to_pylist():
            q = (row.get('question') or '').strip()
            a = (row.get('answer') or '').strip()
            if not q or not a:
                continue
            titles = row['context']['title']
            sents = row['context']['sentences']
            order = list(range(len(titles)))
            rng.shuffle(order)
            order = order[:n_docs]

            blocks = []
            for idx in order:
                text = ' '.join(sents[idx]).strip()
                if not text:
                    continue
                blocks.append(f'[{len(blocks) + 1}] {titles[idx]}: {text}')
            if not blocks:
                continue

            user = ('Read the passages below and answer the question.\n\n'
                    + '\n\n'.join(blocks)
                    + f'\n\nQuestion: {q}\nAnswer with the shortest possible phrase.')
            out.write(json.dumps({'data': [user, a]}, ensure_ascii=False) + '\n')
            kept += 1
            if kept >= limit:
                break
    print(f'hotpotqa: {kept} examples from {src} -> {dst}')


def main():
    args = parse_args()
    if args.longalign:
        convert_longalign(args.longalign, args.out_longalign)
    if args.hotpotqa:
        convert_hotpotqa(args.hotpotqa, args.out_hotpotqa, args.max_examples,
                         args.docs, args.seed)
    if not args.longalign and not args.hotpotqa:
        raise SystemExit('nothing to do: pass --longalign and/or --hotpotqa')


if __name__ == '__main__':
    main()
