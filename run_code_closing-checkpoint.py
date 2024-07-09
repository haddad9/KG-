import pandas as pd
import time
from tqdm import tqdm

import sys
model_name = sys.argv[1]
cuda = sys.argv[2]
output_file = sys.argv[3]

# +
# model_name = 'microsoft/Phi-3-mini-128k-instruct'
# model_name = 'microsoft/Phi-3-small-8k-instruct'
# model_name = 'google/codegemma-7b-it'
# cuda = '2'
# output_file = 'fewshot_closing_gemma.csv'
# -

import os
os.environ['CUDA_VISIBLE_DEVICES'] = cuda

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

from huggingface_hub import login
login(token = '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')

model = AutoModelForCausalLM.from_pretrained(
    model_name, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
).half()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1000,
    "return_full_text": False,
    "do_sample": False,
}

file_path_closing_1shot = './prompts/closing_code_1shot.txt'
with open(file_path_closing_1shot, 'r') as file:
    prompt_closing_1shot = file.read()

file_path_closing_2shot = './prompts/closing_code_2shot.txt'
with open(file_path_closing_2shot, 'r') as file:
    prompt_closing_2shot = file.read()

from datasets import load_from_disk
test_dataset_name = "dataset-surface-info/new-dataset/new-closing/new-closing-1"
test_dataset = load_from_disk(test_dataset_name)

results = pd.DataFrame()
results = pd.DataFrame(columns=['regulatory', '1', '2'])

cnt = 0
for i in tqdm(range(len(test_dataset))[:]):    
    regulatory = test_dataset[i]['regulatory']
    teks = '\n'.join([line for line in test_dataset[i]['text'].splitlines() if line.strip() != ''])
    
    results.at[i, 'regulatory'] = regulatory
    
    print(cnt, regulatory)
    start = time.time()

    messages_1shot = [
        {"role": "user", "content": prompt_closing_1shot.format(regulatory=regulatory,
                                                                text=teks)},
    ]

    try:
        output = pipe(messages_1shot, **generation_args)
        results.at[i, '1'] = output[0]['generated_text']
    except Exception as e:
        results.at[i, '1'] = ''
        print(e)

    end_1 = time.time()
    print(f"1 shot: {end_1-start}")

    messages_2shot = [
        {"role": "user", "content": prompt_closing_2shot.format(regulatory=regulatory,
                                                                text=teks)},
    ]

    try:
        output = pipe(messages_2shot, **generation_args)
        results.at[i, '2'] = output[0]['generated_text']
    except Exception as e:
        results.at[i, '2'] = ''
        print(e)

    end_2 = time.time()
    print(f"2 shot: {end_2-end_1}")

    end = time.time()
    print(regulatory, end-start)
    
    cnt += 1
    
    if cnt % 100 == 0:
        results.to_csv(f'./results/{output_file}.csv')

results.to_csv(f'./results/{output_file}.csv')
