import os
import sys
import json
import pathlib
import pyarrow.dataset as pds
from transformers import GPT2Tokenizer

# a simple parquet-to-uint16le pretokenizing script
# usage: python3 ./prepare_parquet.py [name] [path] [column]
# e.g. "python3 ./prepare_parquet.py cosmopediav2 /home/bctnry/datasets/CosmopediaV2 text"
# remember to use HF_HUB_OFFLINE=1 if you've downloaded the tokenizer already.

Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
Tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})

ds_name = sys.argv[1]
ds_path = sys.argv[2]
ds_path = pathlib.Path(ds_path)
column = '' if len(sys.argv) < 4 else sys.argv[3]
column = column.strip()

ds = pds.dataset(ds_path, format='parquet')
total = ds.count_rows()
print(f'total: {total}')
batches = ds.to_batches(columns=[column])

file_count = 0
EOT = Tokenizer.encode('<|endoftext|>')[0]
current_file = open(f'{ds_name}_{column}_{file_count:08}.bin', 'wb')
batch_count = 0
count = 0
for bb in batches:
    batch_count += 1
    print(f'batch {batch_count}')
    for i, b in enumerate(bb):
        column_name = columns[i]
        for c in b:
            encoded = Tokenizer.encode(str(c))
            encoded.append(EOT)
            for tok in encoded:
                byte1 = tok%256
                byte2 = (tok//256)%256
                current_file.write(bytes([byte1, byte2]))
            count += 1
            if count % 100 == 0:
                print('.', end='', flush=True)
            if count % 1000 == 0:
                print(f'{count / total * 100:.4f}', flush=True)
            if count % 1000000 == 0:
                current_file.close()
                file_count += 1
                current_file = open(f'{ds_name}_{column}_{file_count:08}.bin', 'wb')

current_file.close()


