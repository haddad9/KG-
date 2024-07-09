import pandas as pd
import time
from tqdm import tqdm

import sys
model_name = sys.argv[1]
cuda = sys.argv[2]
name = sys.argv[3]
# start_idx = int(sys.argv[4])

# +
# model_name = 'codellama/CodeLlama-7b-Instruct-hf'
# model_name = 'microsoft/Phi-3-small-8k-instruct'
# model_name = 'google/codegemma-7b-it'
# cuda = '1'
# output_file = 'fewshot_opening_code_llama.csv'
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
# body
file_path_1shot = './prompts/body_struktur_code_1shot.txt'
with open(file_path_1shot, 'r') as file:
    bc_prompt_1shot = file.read()
    
file_path_2shot = './prompts/body_struktur_code_2shot.txt'
with open(file_path_2shot, 'r') as file:
    bc_prompt_2shot = file.read()
    
file_path_1shot = './prompts/body_struktur_1shot.txt'
with open(file_path_1shot, 'r') as file:
    bf_prompt_1shot = file.read()
    
file_path_2shot = './prompts/body_struktur_2shot.txt'
with open(file_path_2shot, 'r') as file:
    bf_prompt_2shot = file.read()
    
# closing    
file_path_1shot = './prompts/closing_code_1shot.txt'
with open(file_path_1shot, 'r') as file:
    cc_prompt_1shot = file.read()
    
file_path_2shot = './prompts/closing_code_2shot.txt'
with open(file_path_2shot, 'r') as file:
    cc_prompt_2shot = file.read()
    
file_path_1shot = './prompts/closing_1shot.txt'
with open(file_path_1shot, 'r') as file:
    cf_prompt_1shot = file.read()
    
file_path_2shot = './prompts/closing_2shot.txt'
with open(file_path_2shot, 'r') as file:
    cf_prompt_2shot = file.read()
    
# opening
file_path_1shot = './prompts/opening_code_1shot_1.txt'
with open(file_path_1shot, 'r') as file:
    oc_prompt_1shot_1 = file.read()

file_path_1shot = './prompts/opening_code_1shot_2.txt'
with open(file_path_1shot, 'r') as file:
    oc_prompt_1shot_2 = file.read()
    
file_path_2shot = './prompts/opening_code_2shot_1.txt'
with open(file_path_2shot, 'r') as file:
    oc_prompt_2shot_1 = file.read()
    
file_path_2shot = './prompts/opening_code_2shot_2.txt'
with open(file_path_2shot, 'r') as file:
    oc_prompt_2shot_2 = file.read()
    
file_path_1shot = './prompts/opening_1shot_1.txt'
with open(file_path_1shot, 'r') as file:
    of_prompt_1shot_1 = file.read()
    
file_path_1shot = './prompts/opening_1shot_2.txt'
with open(file_path_1shot, 'r') as file:
    of_prompt_1shot_2 = file.read()
    
file_path_2shot = './prompts/opening_2shot_1.txt'
with open(file_path_2shot, 'r') as file:
    of_prompt_2shot_1 = file.read()
    
file_path_2shot = './prompts/opening_2shot_2.txt'
with open(file_path_2shot, 'r') as file:
    of_prompt_2shot_2 = file.read()

# +
from datasets import load_from_disk

b = load_from_disk("dataset-surface-info/new-new-body-struktur/new-new-body-struktur-1")
o = load_from_disk("dataset-surface-info/new-new-opening/new-new-opening-1")
c = load_from_disk("dataset-surface-info/new-new-closing/new-new-closing-1")


# +
def kgc_1(part, model, test_dataset, prompt_1shot, prompt_2shot):
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
            {"role": "user", "content": prompt_1shot.format(regulatory=regulatory,
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
            {"role": "user", "content": prompt_2shot.format(regulatory=regulatory,
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

#         if cnt % 10 == 0:
#             results.to_csv(f'./results/{part}_{model}.csv')

#     results.to_csv(f'./results/{part}_{model}.csv')


# +
def kgc_2(part, model, test_dataset, prompt_opening_1shot_1, prompt_opening_1shot_2, prompt_opening_2shot_1, prompt_opening_2shot_2):
    results = pd.DataFrame()
    results = pd.DataFrame(columns=['regulatory', '1_1', '1_2', '2_1', '2_2'])

    cnt = 0
    for i in tqdm(range(len(test_dataset))[:]):
        regulatory = test_dataset[i]['regulatory']
        teks_1 = '\n'.join([line for line in test_dataset[i]['text_1'].splitlines() if line.strip() != ''])
        teks_2 = '\n'.join([line for line in test_dataset[i]['text_2'].splitlines() if line.strip() != ''])

        results.at[i, 'regulatory'] = regulatory

        print(cnt, regulatory)
        start = time.time()

        ### 1 shot bagian 1 (Mengingat)
        messages_1shot_1 = [
            {"role": "user", "content": prompt_opening_1shot_1.format(regulatory=regulatory,
                                                                      text=teks_1)},
        ]

        try:
            output = pipe(messages_1shot_1, **generation_args)
            results.at[i, '1_1'] = output[0]['generated_text']
        except Exception as e:
            results.at[i, '1_1'] = ''
            print(e)

        end_1 = time.time()
        print(f"1 shot bagian 1: {end_1-start}")

        ### 1 shot bagian 2 (Selain Mengingat)
        messages_1shot_2 = [
            {"role": "user", "content": prompt_opening_1shot_2.format(regulatory=regulatory,
                                                                      text=teks_2)},
        ]

        try:
            output = pipe(messages_1shot_2, **generation_args)
            results.at[i, '1_2'] = output[0]['generated_text']
        except Exception as e:
            results.at[i, '1_2'] = ''
            print(e)

        end_2 = time.time()
        print(f"1 shot bagian 2: {end_2-start}")

        ### 2 shot bagian 1 (Mengingat)
        messages_2shot_1 = [
            {"role": "user", "content": prompt_opening_2shot_1.format(regulatory=regulatory,
                                                                      text=teks_1)},
        ]

        try:
            output = pipe(messages_2shot_1, **generation_args)
            results.at[i, '2_1'] = output[0]['generated_text']
        except Exception as e:
            results.at[i, '2_1'] = ''
            print(e)

        end_3 = time.time()
        print(f"2 shot bagian 1: {end_3-end_2}")

        ### 2 shot bagian 2 (Selain Mengingat)
        messages_2shot_2 = [
            {"role": "user", "content": prompt_opening_2shot_2.format(regulatory=regulatory,
                                                                      text=teks_2)},
        ]

        try:
            output = pipe(messages_2shot_2, **generation_args)
            results.at[i, '2_2'] = output[0]['generated_text']
        except Exception as e:
            results.at[i, '2_2'] = ''
            print(e)

        end_4 = time.time()
        print(f"2 shot bagian 2: {end_4-end_3}")

        end = time.time()
        print(regulatory, end-start)

        cnt += 1

#         if cnt % 10 == 0:
#             results.to_csv(f'./results/{part}_{model}.csv')
            
#     results.to_csv(f'./results/{part}_{model}.csv')


# -

kgc_1('cf', name, c, cf_prompt_1shot, cf_prompt_2shot)
kgc_1('cc', name, c, cc_prompt_1shot, cc_prompt_2shot)
kgc_1('bf', name, b, bf_prompt_1shot, bf_prompt_2shot)
kgc_1('bc', name, b, bc_prompt_1shot, bc_prompt_2shot)
kgc_2('of', name, o, of_prompt_1shot_1, of_prompt_1shot_2, of_prompt_2shot_1, of_prompt_2shot_2)
kgc_2('oc', name, o, oc_prompt_1shot_1, oc_prompt_1shot_2, oc_prompt_2shot_1, oc_prompt_2shot_2)
