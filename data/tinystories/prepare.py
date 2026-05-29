# data/tinystories/prepare.py
import os
import numpy as np
import tiktoken
from tqdm import tqdm
from datasets import load_dataset  # huggingface datasets

# Number of parallel workers for Hugging Face map function
num_proc = 8

# Use standard GPT-2 BPE encoding
enc = tiktoken.get_encoding("gpt2")

def process(example):
    # encode_ordinary ignores any special tokens (like <|endoftext|>) during encoding
    ids = enc.encode_ordinary(example['text'])
    # Append standard <|endoftext|> token (ID: 50256) so the model knows where stories end
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out

if __name__ == '__main__':
    # 1. Download the TinyStories dataset from Hugging Face
    print("Downloading roneneldan/TinyStories dataset from Hugging Face...")
    dataset = load_dataset("roneneldan/TinyStories")

    # 2. Tokenize the split data
    print("Tokenizing data splits using GPT-2 BPE...")
    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # 3. Export tokenized arrays directly to binary files
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        
        # Since GPT-2's vocab is 50,257, uint16 (0 - 65,535) is the perfect datatype
        dtype = np.uint16 
        
        print(f"Writing {filename} (Total tokens: {arr_len:,})...")
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        # We batch write in blocks of 50,000 to keep memory footprint very light
        step_size = 50000
        idx = 0
        for i in tqdm(range(0, len(dset), step_size), desc=f"Writing {split}"):
            chunk = dset.select(range(i, min(i + step_size, len(dset))))
            chunk_ids = np.concatenate([x for x in chunk['ids']])
            arr[idx : idx + len(chunk_ids)] = chunk_ids
            idx += len(chunk_ids)
            
        arr.flush()
        print(f"Finished writing {filename}!")
