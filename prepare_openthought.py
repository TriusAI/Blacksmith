import os
import sys
import json
import pathlib
import pyarrow.dataset as pds
from transformers import GPT2Tokenizer
from tokenmaker import Tokenizer, TokenizerNoThinking

# this script is for open-thoughts/OpenThoughts-114k
# https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k/tree/main/metadata

# HF_HUB_OFFLINE=1

# we use the following format:
#     <|im_start|>user
#     {user prompt...}<|im_end|>
#     <|im_start|>assistant
#     <think>{thinking content...}</think>
#     {assistant answer...}<|im_end|>

ds_name = 'openthought'
ds_path = '/home/bctnry/workspace/datasets/OpenThoughts-114k/data'
ds_path = pathlib.Path(ds_path)
columns = ['problem','deepseek_reasoning','deepseek_solution']

IM_START = Tokenizer.encode('<|im_start|>')[0]
IM_END = Tokenizer.encode('<|im_end|>')[0]
EOT = Tokenizer.eos_token_id  # 50256
NEWLINE = Tokenizer.encode('\n')[0]
USER = Tokenizer.encode('user')
ASSISTANT = Tokenizer.encode('assistant')
THINK_START = Tokenizer.encode('<think>')[0]
THINK_END = Tokenizer.encode('</think>')[0]

ds = pds.dataset(ds_path, format='parquet')
total = ds.count_rows()
print(f'total: {total}')
batches = ds.to_batches(columns=columns)

file_count = 0
current_file = open(f'{ds_name}_{file_count:08}.bin', 'wb')
batch_count = 0
count = 0
for bb in batches:
    batch_count += 1
    print(f'batch {batch_count}')
    problems = bb.column('problem')
    reasonings = bb.column('deepseek_reasoning')
    answers = bb.column('deepseek_solution')
    bblen = len(bb)
    for i in range(bblen):
        problem = problems[i]
        deepseek_reasoning = reasonings[i]
        deepseek_solution = answers[i]
        encoded = [IM_START] + USER + [NEWLINE]
        encoded += TokenizerNoThinking.encode(str(problem))
        encoded += [IM_END, NEWLINE]
        encoded += [IM_START] + ASSISTANT + [NEWLINE]
        encoded += [THINK_START]
        encoded += TokenizerNoThinking.encode(str(deepseek_reasoning))
        encoded += [THINK_END, NEWLINE]
        encoded += TokenizerNoThinking.encode(str(deepseek_solution))
        encoded += [IM_END]
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


