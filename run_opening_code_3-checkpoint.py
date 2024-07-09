import pandas as pd
import time
from tqdm import tqdm

import sys
model_name = sys.argv[1]
cuda = sys.argv[2]
output_file = sys.argv[3]
start_idx = int(sys.argv[4])

# +
# model_name = 'codellama/CodeLlama-7b-Instruct-hf'
# model_name = 'microsoft/Phi-3-small-8k-instruct'
# model_name = 'google/codegemma-7b-it'
# model_name = 'codellama/CodeLlama-7b-Instruct-hf'
# cuda = '2'
# output_file = 'fewshot_opening_code_phi_label'
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
    "max_new_tokens": 2000,
    "return_full_text": False,
    "do_sample": False,
}

# +
file_path_opening_1shot_1 = './prompts/opening_code_1shot_3.txt'
with open(file_path_opening_1shot_1, 'r') as file:
    prompt_opening_1shot_1 = file.read()

# file_path_opening_1shot_2 = './prompts/opening_code_1shot_2.txt'
# with open(file_path_opening_1shot_2, 'r') as file:
#     prompt_opening_1shot_2 = file.read()

# +
file_path_opening_2shot_1 = './prompts/opening_code_2shot_3.txt'
with open(file_path_opening_2shot_1, 'r') as file:
    prompt_opening_2shot_1 = file.read()

# file_path_opening_2shot_2 = './prompts/opening_code_2shot_2.txt'
# with open(file_path_opening_2shot_2, 'r') as file:
#     prompt_opening_2shot_2 = file.read()
# -

from datasets import load_from_disk
test_dataset_name = "dataset-surface-info/new-opening/new-opening-1"
test_dataset = load_from_disk(test_dataset_name)

results = pd.DataFrame()
results = pd.DataFrame(columns=['regulatory', '1_3', '2_3'])

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

# +
cnt = 0
for i in tqdm(range(len(test_dataset))[start_idx:]):
    regulatory = test_dataset[i]['regulatory']
    teks_1 = '\n'.join([line for line in test_dataset[i]['text_1'].splitlines() if line.strip() != ''])
    teks_2 = '\n'.join([line for line in test_dataset[i]['text_2'].splitlines() if line.strip() != ''])

    results.at[i, 'regulatory'] = regulatory

    print(cnt, regulatory)
    start = time.time()

    ### 1 shot bagian 1 (Mengingat)
    messages_1shot_1 = [
        {"role": "user", "content": prompt_opening_1shot_1.format(regulatory=regulatory,
                                                                  text=teks_2)},
    ]

    try:
        output = pipe(messages_1shot_1, **generation_args)
        results.at[i, '1_3'] = output[0]['generated_text']
    except Exception as e:
        results.at[i, '1_3'] = ''
        print(e)

    end_1 = time.time()
    print(f"1 shot bagian 1: {end_1-start}")

#     ### 1 shot bagian 2 (Selain Mengingat)
#     messages_1shot_2 = [
#         {"role": "user", "content": prompt_opening_1shot_2.format(regulatory=regulatory,
#                                                                   text=teks_2)},
#     ]

#     try:
#         output = pipe(messages_1shot_2, **generation_args)
#         results.at[i, '1_2'] = output[0]['generated_text']
#     except Exception as e:
#         results.at[i, '1_2'] = ''
#         print(e)

#     end_2 = time.time()
#     print(f"1 shot bagian 2: {end_2-start}")

    ### 2 shot bagian 1 (Mengingat)
    messages_2shot_1 = [
        {"role": "user", "content": prompt_opening_2shot_1.format(regulatory=regulatory,
                                                                  text=teks_2)},
    ]

    try:
        output = pipe(messages_2shot_1, **generation_args)
        results.at[i, '2_3'] = output[0]['generated_text']
    except Exception as e:
        results.at[i, '2_3'] = ''
        print(e)

    end_3 = time.time()
    print(f"2 shot bagian 1: {end_3-end_1}")

#     ### 2 shot bagian 2 (Selain Mengingat)
#     messages_2shot_2 = [
#         {"role": "user", "content": prompt_opening_2shot_2.format(regulatory=regulatory,
#                                                                   text=teks_2)},
#     ]

#     try:
#         output = pipe(messages_2shot_2, **generation_args)
#         results.at[i, '2_2'] = output[0]['generated_text']
#     except Exception as e:
#         results.at[i, '2_2'] = ''
#         print(e)

#     end_4 = time.time()
#     print(f"2 shot bagian 2: {end_4-end_3}")

    end = time.time()
    print(regulatory, end-start)

    cnt += 1

    if cnt % 10 == 0:
        results.to_csv(f'./results/{output_file}.csv')
# -

results.to_csv(f'./results/{output_file}.csv')
