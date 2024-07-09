import pandas as pd
import time
from tqdm import tqdm

import sys
# model_name = sys.argv[1]
# cuda = sys.argv[2]
# output_file = sys.argv[3]

# !nvidia-smi

# model_name = 'microsoft/Phi-3-mini-128k-instruct'
# model_name = 'microsoft/Phi-3-small-8k-instruct'
# model_name = 'google/codegemma-7b-it'
model_name = "bigscience/bloomz-7b1"
cuda = '2'
# output_file = 'fewshot_closing_gemma.csv'

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

# +
file_path_closing_1shot = './prompts/closing_1shot.txt'
with open(file_path_closing_1shot, 'r') as file:
    prompt_closing_1shot = file.read()

file_path_closing_2shot = './prompts/closing_2shot.txt'
with open(file_path_closing_2shot, 'r') as file:
    prompt_closing_2shot = file.read()
# -

from datasets import load_from_disk
test_dataset_name = "dataset-surface-info/new-dataset/new-closing/new-closing-1"
test_dataset = load_from_disk(test_dataset_name)

# +
regulatory = test_dataset[0]['regulatory']
teks = '\n'.join([line for line in test_dataset[0]['text'].splitlines() if line.strip() != ''])

start = time.time()

messages_1shot = [
    {"role": "user", "content": prompt_closing_1shot.format(regulatory=regulatory,
                                                            text=teks)},
]

try:
    output = pipe(messages_1shot, **generation_args)
    print(output)
except Exception as e:
    print(e)

end_1 = time.time()
print(f"1 shot: {end_1-start}")

messages_2shot = [
    {"role": "user", "content": prompt_closing_2shot.format(regulatory=regulatory,
                                                            text=teks)},
]

try:
    output = pipe(messages_2shot, **generation_args)
    print(output)
except Exception as e:
    print(e)

end_2 = time.time()
print(f"2 shot: {end_2-end_1}")

end = time.time()
print(regulatory, end-start)
# -

print(output[0]['generated_text'])

generation_config_f7 = model.generation_config
generation_config_f7.temperature = 0.0
generation_config_f7.max_new_tokens = 3000
generation_config_f7.pad_token_id = tokenizer.eos_token_id
generation_config_f7.eos_token_id = tokenizer.eos_token_id
generation_config_f7

# +
regulatory = test_dataset[326]['regulatory']
teks = '\n'.join([line for line in test_dataset[326]['text'].splitlines() if line.strip() != ''])

test = prompt_closing_1shot.format(regulatory=regulatory, text=teks)
input_ids = tokenizer(test, return_tensors="pt").input_ids
input_ids = input_ids.to(model.device)

# +
# %%time

with torch.inference_mode():
    outputs = model.generate(
        input_ids = input_ids,
        generation_config = generation_config_f7,
    )
# -

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)



results = pd.DataFrame()
results = pd.DataFrame(columns=['regulatory', '1', '2'])

# +
# results.at[0, 'regulatory'] = test_dataset[100]['regulatory']

# regulatory = test_dataset[0]['regulatory']
# teks_1 = '\n'.join([line for line in test_dataset[100]['text_1'].splitlines() if line.strip() != ''])
# teks_2 = '\n'.join([line for line in test_dataset[100]['text_2'].splitlines() if line.strip() != ''])

# ### 1 shot bagian 1 (Mengingat)
# messages_1shot_1 = [
#     {"role": "user", "content": prompt_opening_1shot_1.format(regulatory=regulatory,
#                                                               text=teks_1)},
# ]

# try:
#     output = pipe(messages_1shot_1, **generation_args)
#     results.at[0, '1_1'] = output[0]['generated_text']
#     print(output[0]['generated_text'])
#     print()
#     print(teks_1)
#     print()
# except Exception as e:
#     results.at[0, '1_2'] = ''
#     print(e)
#     print()
    
# end_1 = time.time()
# print(f"1 shot bagian 1: {end_1-start}")
# print()

# ### 1 shot bagian 2 (Selain Mengingat)
# messages_1shot_2 = [
#     {"role": "user", "content": prompt_opening_1shot_2.format(regulatory=regulatory,
#                                                               text=teks_2)},
# ]

# try:
#     output = pipe(messages_1shot_2, **generation_args)
#     results.at[0, '1_2'] = output[0]['generated_text']
#     print(output[0]['generated_text'])
#     print()
#     print(teks_2)
#     print()
# except Exception as e:
#     results.at[0, '1_2'] = ''
#     print(e)
#     print()
    
# end_2 = time.time()
# print(f"1 shot bagian 2: {end_2-start}")
# print()    

# ### 2 shot bagian 1 (Mengingat)
# messages_2shot_1 = [
#     {"role": "user", "content": prompt_opening_2shot_1.format(regulatory=regulatory,
#                                                               text=teks_1)},
# ]

# try:
#     output = pipe(messages_2shot_1, **generation_args)
#     results.at[0, '2_1'] = output[0]['generated_text']
#     print(output[0]['generated_text'])
#     print()
#     print(teks_1)
#     print()
# except Exception as e:
#     results.at[0, '2_1'] = ''
#     print(e)
#     print()
    
# end_3 = time.time()
# print(f"2 shot bagian 1: {end_3-end_2}")
# print()

# ### 2 shot bagian 2 (Selain Mengingat)
# messages_2shot_2 = [
#     {"role": "user", "content": prompt_opening_2shot_2.format(regulatory=regulatory,
#                                                               text=teks_2)},
# ]

# try:
#     output = pipe(messages_2shot_2, **generation_args)
#     results.at[0, '2_2'] = output[0]['generated_text']
#     print(output[0]['generated_text'])
#     print()
#     print(teks_2)
#     print()
# except Exception as e:
#     results.at[0, '2_1'] = ''
#     print(e)
#     print()
    
# end_4 = time.time()
# print(f"2 shot bagian 2: {end_4-end_3}")
# print()   
# -

cnt = 0
for i in tqdm(range(len(test_dataset))[:]):    
    regulatory = test_dataset[i]['regulatory']
    teks = '\n'.join([line for line in test_dataset[i]['text'].splitlines() if line.strip() != ''])
    
    results.at[0, 'regulatory'] = regulatory
    
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
