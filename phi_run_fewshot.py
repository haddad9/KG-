import pandas as pd
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)

model_phi = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer_phi = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

pipe_phi = pipeline(
    "text-generation",
    model=model_phi,
    tokenizer=tokenizer_phi,
)

generation_args = {
    "max_new_tokens": 1000,
    "return_full_text": False,
    "do_sample": False,
}

df_fewshot = pd.read_csv('./dataset-fewshot/closing.csv')


from datasets import load_from_disk

test_dataset_name = "dataset-surface-info/split/closing/test"
test_dataset = load_from_disk(test_dataset_name)

prompt_few_shot = """Anda adalah pembuat knowledge graph. Anda akan diberikan entity regulatory dan teks, dari teks tersebut lakukan tindakan berikut:
- identifikasi sebanyak mungkin relation di antara entity
- keluarkan daftar dalam format triple turtle ENTITY 1 RELATION 1 OBJECT 1/ENTITY 2 .

Tipe entity paling penting adalah:
- lexid-s:LegalDocument, contoh: lexid:Permen_Agama_2019_17
- lexid-s:Person, contoh: lexid:Person_Joko_Widodo
- lexid-s:Position, contoh: lexid:Position_Menteri_Agama_Republik_Indonesia
- lexid-s:City, contoh: lexid:City_Depok

Relation paling penting adalah:
- lexid-s:hasEnactionDate, yaitu kapan peraturan tersebut ditetapkan, contoh: lexid:PP_2019_4 lexid-s:hasEnactionDate "2019-01-28"^^xsd:date.
- lexid-s:hasEnactionLocation, yaitu dimana peraturan tersebut ditetapkan, contoh: lexid:PP_2019_4 lexid-s:hasEnactionLocation lexid:City_Jakarta.
- lexid-s:hasEnactionOfficial, yaitu siapa nama orang yang menetapkan peraturan tersebut, contoh: lexid:PP_2019_4 lexid-s:hasEnactionOfficial lexid:Person_Joko_Widodo.
- lexid-s:hasEnactionOffice, yaitu apa jabatan dari orang yang menetapkan peraturan tersebut, contoh: lexid:PP_2019_4 lexid-s:hasEnactionOffice lexid:Position_Presiden_Republik_Indonesia.
- lexid-s:hasPromulgationDate, yaitu kapan peraturan tersebut diudangkan, contoh: lexid:PP_2019_4 lexid-s:hasPromulgationDate "2019-01-28"^^xsd:date.
- lexid-s:hasPromulgationLocation, yaitu dimana peraturan tersebut diundangkan, contoh: lexid:PP_2019_4 lexid-s:hasPromulgationLocation lexid:City_Jakarta.
- lexid-s:hasPromulgationOfficial, yaitu siapa nama orang yang mengundangkan peraturan tersebut, contoh: lexid:PP_2019_4 lexid-s:hasPromulgationOfficial lexid:Person_Yasonna_H_Laolly.
- lexid-s:hasPromulgationOffice, yaitu apa jabatan dari orang yang mengundangkan peraturan tersebut, contoh: lexid:PP_2019_4 lexid-s:hasPromulgationOffice lexid:Position_Menteri_Hukum_Dan_Hak_Asasi_Manusia_Republik_Indonesia.
- lexid-s:hasPromulgationPlace, yaitu dimana pengundangan peraturan tersebut ditempatkan, contoh: lexid:PP_2019_4 lexid-s:hasPromulgationPlace lexid:Lembaran_Negara.

JANGAN UBAH APAPUN YANG ADA PADA TEKS. GUNAKAN BAHASA INDONESIA SEPERTI YANG ADA PADA TEKS.

Contoh:
"""

prompt_regulatory = """
###regulatory: lexid:{regulatory}
###Teks: 
```
{text}
```
###output:

"""

def get_prompt_fewshot(k_shot):
    prompt = prompt_few_shot
    
    for i in range(k_shot):
        prompt += prompt_regulatory.format(regulatory = df_fewshot.iloc[i]['regulatory'], text=df_fewshot.iloc[i]['text'])
        prompt += df_fewshot.iloc[i]['triples'] + '\n' 
    
    return prompt

results = pd.DataFrame()
results = pd.DataFrame(columns=['regulatory', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'])

cnt = 0
for i in tqdm(range(len(test_dataset))[:]):
    results.at[i, 'regulatory'] = test_dataset[i]['regulatory']
    for j in tqdm(range(1,12)[:]):
        prompt =  get_prompt_fewshot(j)
        prompt += prompt_regulatory.format(regulatory = test_dataset[i]['regulatory'], text=test_dataset[i]['text'])
        messages = [
            {"role": "user", "content": prompt},
        ]
        
        try:
            output = pipe_phi(messages, **generation_args)
            results.at[i, f'{j}'] = output[0]['generated_text']
        except:
            results.at[i, f'{j}'] = ''
    
    cnt += 1
    
    if cnt % 100 == 0:
        results.to_csv('./result/fewshot_closing.csv')

results.to_csv('./result/fewshot_closing.csv')
