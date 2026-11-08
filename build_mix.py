#!/usr/bin/env python3
"""Build a mixed JSONL by sampling each source down to a token budget.

Token shares are what the packed dataset actually sees, so integer `repeat`
factors on whole files are too coarse (LongAlign alone is 265M tokens). This
samples rows per source until each budget is met, then shuffles.

    python3 build_mix.py --out mix_sft.jsonl \
        --source identity.jsonl:1000000 --source train_0.jsonl:28000000 \
        --source longalign_sft.jsonl:9000000 --source niah_sft.jsonl:55000000
"""
import argparse
import json
import random

from tokenmaker import Tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--source', action='append', required=True,
                   metavar='PATH:TOKENS', help='repeatable; tokens is a budget')
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    rows, stats = [], []
    for spec in args.source:
        path, _, budget = spec.rpartition(':')
        budget = int(budget)
        used = 0
        n = 0
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                msgs = d.get('data') or []
                used += sum(len(Tokenizer.encode(m)) + 6 for m in msgs)
                rows.append(d)
                n += 1
                if used >= budget:
                    break
        stats.append((path, n, used))

    rng.shuffle(rows)
    with open(args.out, 'w') as f:
        for d in rows:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')

    total = sum(u for _, _, u in stats)
    for path, n, used in stats:
        print(f'  {path:28s} {n:7d} rows {used/1e6:6.1f}M tokens '
              f'({100*used/total:4.1f}%)')
    print(f'wrote {len(rows)} rows, {total/1e6:.1f}M tokens -> {args.out}')


if __name__ == '__main__':
    main()
