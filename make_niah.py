#!/usr/bin/env python3
"""Generate needle-in-a-haystack training data in the sft_dataset.py JSONL
format.

Each example is one user turn holding a long filler text with one or more
"passphrase" sentences buried at random depths, plus a question; the assistant
turn is the answer. This is the same distribution eval_retrieval.py measures, so
training on it should show up directly in the depth-0.5 column.

    python3 make_niah.py --n 4000 --min-len 1024 --max-len 32768 \
        --keys 1 --out niah_train.jsonl

Filler comes from databricks-dolly-15k.jsonl (already on disk), so no download
is needed; use --filler to point at another JSONL with a "response" field.
"""
import argparse
import json
import random

from tokenmaker import Tokenizer

WORDS = ('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo '
         'lima mike november oscar papa quebec romeo sierra tango uniform victor '
         'whiskey xray yankee zulu').split()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1000)
    p.add_argument('--out', default='niah_train.jsonl')
    p.add_argument('--min-len', type=int, default=1024)
    p.add_argument('--max-len', type=int, default=32768)
    p.add_argument('--keys', type=int, default=1,
                   help='passphrases per document; answers are comma separated')
    p.add_argument('--mode', choices=['named', 'global', 'mixed'], default='mixed',
                   help='named = "The passphrase for X is Y" / "What is the passphrase '
                        'for X?"; global = "The secret passphrase is Y" / "Reminder: '
                        'the secret passphrase is", which is what eval_retrieval.py '
                        'probes. mixed emits both so the wording is not a confound.')
    p.add_argument('--filler', default='databricks-dolly-15k.jsonl')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--pool-tokens', type=int, default=200000)
    return p.parse_args()


def build_pool(path, target):
    """A long run of natural English, tokenised once and reused."""
    parts, total = [], 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line).get('response', '').strip()
            except json.JSONDecodeError:
                continue
            if not r:
                continue
            parts.append(r)
            total += len(Tokenizer.encode(r)) + 2
            if total > target:
                break
    ids = Tokenizer.encode('\n\n'.join(parts))
    if len(ids) < target:
        ids = (ids * (target // max(1, len(ids)) + 1))[:target]
    return ids[:target]


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    pool = build_pool(args.filler, args.pool_tokens)
    print(f'filler pool: {len(pool)} tokens')

    written = 0
    with open(args.out, 'w') as out:
        for _ in range(args.n):
            length = rng.randint(args.min_len, args.max_len)
            body = pool[:min(length, len(pool))]
            if len(body) < length:                     # wrap around if needed
                body = (pool * (length // len(pool) + 1))[:length]

            pairs = []
            for _ in range(args.keys):
                name = rng.choice(WORDS) + '-' + str(rng.randint(10, 99))
                code = rng.choice(WORDS).upper() + '-' + str(rng.randint(100, 999))
                pairs.append((name, code))
            mode = args.mode if args.mode != 'mixed' else rng.choice(['named', 'global'])

            # Insert each needle at an independent random depth, deepest first so
            # earlier offsets stay valid.
            inserts = []
            for name, code in pairs:
                cut = rng.randint(0, len(body))
                if mode == 'global':
                    needle = f'\n\nThe secret passphrase is {code}.\n\n'
                else:
                    needle = f'\n\nThe passphrase for {name} is {code}.\n\n'
                inserts.append((cut, Tokenizer.encode(needle)))
            inserts.sort(key=lambda x: x[0])
            for cut, nid in reversed(inserts):
                body = body[:cut] + nid + body[cut:]

            if args.keys == 1:
                name, code = pairs[0]
                if mode == 'global':
                    question = ('What is the secret passphrase mentioned above? '
                                'Answer with the passphrase only.')
                else:
                    question = (f'What is the passphrase for {name}? '
                                'Answer with the passphrase only.')
                answer = code
            else:
                question = ('List the passphrase for each name mentioned above, '
                            'one "name: passphrase" pair per line.')
                answer = '\n'.join(f'{n}: {c}' for n, c in pairs)

            document = Tokenizer.decode(body)
            user = f'{document}\n\n{question}'
            out.write(json.dumps({'data': [user, answer]}, ensure_ascii=False) + '\n')
            written += 1
            if written % 500 == 0:
                print(f'  {written}/{args.n}', flush=True)

    print(f'wrote {written} examples -> {args.out}')


if __name__ == '__main__':
    main()
