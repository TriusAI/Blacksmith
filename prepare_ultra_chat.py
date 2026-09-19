import os
import json
import pathlib
from transformers import GPT2Tokenizer

# we use the format `<|im_start|>{role}\n{content}<|im_end|>`, e.g.:
#
#     User: What color is the sky?
#     Assistant: Blue, when it's sunny and the sun is not blocked
#                by the clouds.
#
# would be encoded as:
#
#     <|im_start|>user
#     What color is the sky?<|im_end|>
#     <|im_start|>assistant
#     Blue, when it's sunny and the sun is not blocked by the clouds.<|im_end|>

Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
Tokenizer.add_special_tokens({'additional_special_tokens': ['<|im_start|>', '<|im_end|>']})


# script for preparing UltraChat to a single .bin file (so i can memmap)
# the dataset is downloaded from:
#     https://huggingface.co/datasets/openbmb/UltraChat/tree/main
# which is in .jsonl.

# if you have downloaded the gpt2 tokenizer already it's strongly
# recommended that you run this with HF_HUB_OFFLINE=1 because those
# rat bastards just have to make their library poke the internet with
# its HTTP client dick despite having downloaded the fucking thing to
# local cache.

dataset_path = pathlib.Path('/home/bctnry/workspace/datasets/UltraChat/')
result = open('UltraChat_uint16le.bin', 'wb')
failed = []

EOT = Tokenizer.eos_token_id

def write_tokens(tokens):
    for tok in tokens:
        byte1 = tok%256
        byte2 = (tok//256)%256
        result.write(bytes([byte1, byte2]))

for file_i in range(0, 10):   # files: train_0 ~ train_9
    print(f'processing file `train_{file_i}.jsonl`...')
    with open(dataset_path / f'train_{file_i}.jsonl', 'r') as f:
        i = 0
        while True:
            a = f.readline()
            if not a.strip(): break
            try:
                j = json.loads(a)
            except:
                failed.append((a, file_i, i))
                continue
            for ii, d in enumerate(j['data']):
                role = 'user' if ii % 2 == 0 else 'assistant'
                s = f'<|im_start|>{role}\n{d}<|im_end|>\n'
                encoded = Tokenizer.encode(s)
                write_tokens(encoded)
            write_tokens([EOT])
            i += 1
            if i % 100 == 0:
                print('.', end='', flush=True)
            if i % 10000 == 0:
                print(f' file {file_i}: {i} conversations', flush=True)
    
result.close()
with open('failed_items.txt', 'w') as f:
    for k, _, _ in failed:
        f.write(k)
        f.write('\n')
with open('failed_item_location.txt', 'w') as f:
    for _, file_i, i in failed:
        f.write(f'file {file_i} line {i}\n')
        


