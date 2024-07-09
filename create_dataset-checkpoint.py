# +
import os
import pandas as pd
import time
import re

# from fast_ml.model_development import train_valid_test_split
from sklearn.model_selection import train_test_split

# +
import os

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

bn_path = 'new_parsed_files/bn'
ln_path = 'new_parsed_files/ln'
lain_path = 'new_parsed_files/lain-lain'
perda_path = 'new_parsed_files/perda'
putusan_path = 'new_parsed_files/putusan'

bn_count = count_files(bn_path)
ln_count = count_files(ln_path)
lain_count = count_files(lain_path)
perda_count = count_files(perda_path)
putusan_count = count_files(putusan_path)

total_files = bn_count + ln_count + lain_count + perda_count + putusan_count

print(f"Number of files: {total_files}")


# +
import os

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

bn_path = 'new_1_text_files/bn'
ln_path = 'new_1_text_files/ln'
lain_path = 'new_1_text_files/lain-lain'
perda_path = 'new_1_text_files/perda'
putusan_path = 'new_1_text_files/putusan'

bn_count = count_files(bn_path)
ln_count = count_files(ln_path)
lain_count = count_files(lain_path)
perda_count = count_files(perda_path)
putusan_count = count_files(putusan_path)

total_files = bn_count + ln_count + lain_count + perda_count + putusan_count

print(f"Number of files: {total_files}")

# +
import os

def count_files(directory):
    file_count = 0
    for root, dirs, files in os.walk(directory):
        file_count += len(files)
    return file_count

bn_path = 'new_turtle_files/bn'
ln_path = 'new_turtle_files/ln'
lain_path = 'new_turtle_files/lain-lain'
perda_path = 'new_turtle_files/perda'
putusan_path = 'new_turtle_files/putusan'

bn_count = count_files(bn_path)
ln_count = count_files(ln_path)
lain_count = count_files(lain_path)
perda_count = count_files(perda_path)
putusan_count = count_files(putusan_path)

total_files = bn_count + ln_count + lain_count + perda_count + putusan_count

print(f"Number of files: {total_files}")
# -

from datasets import Dataset, load_from_disk

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# # Surface (Opening, Closing, Body)

# 1. baca regulatory map
# 2. buat kolom label
# 3. exclude: label < 4, amandemen
# 4. buat kolom text & triples

# +
def list_files_in_directory(directory):
    files_list = []

    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)

        if os.path.isfile(full_path):
            files_list.append(entry)

    return files_list

def extract_number(filename):
    match = re.search(r'-?\d+', filename)
    return int(match.group()) if match else None

def extract_non_number(filename):
    match = re.search(r'_', filename)
    return match.group() if match else None

def find_opening_closing(files):
    numbered_files = [(extract_number(file), file) for file in files if (extract_number(file)) != None]
    
    opening_file_tuple = min(numbered_files, key=lambda x: x[0])
    closing_file_tuple = max(numbered_files, key=lambda x: x[0])
    
    numbered_files.remove(opening_file_tuple)
    opening_file_1_tuple = min(numbered_files, key=lambda x: x[0])
    
    numbered_files.remove(opening_file_1_tuple)
    opening_file_2_tuple = min(numbered_files, key=lambda x: x[0])
    
    opening_file = opening_file_tuple[1]
    opening_file_1 = opening_file_1_tuple[1]
    opening_file_2 = opening_file_2_tuple[1]
    closing_file = closing_file_tuple[1]
    
    return opening_file_2, opening_file_1, opening_file, closing_file

def find_body_struktur(files):
    non_number_files = [(extract_non_number(file), file) for file in files if (extract_non_number(file)) != None]
    non_number_files = non_number_files[0][1]
    return non_number_files


# +
# tes, tes2, tes3, tes4 = find_opening_closing(list_files_in_directory('split_txt/ln/2019/pp4-2019bt'))
# print(tes, tes2, tes3, tes4)

tes, tes2, tes3, tes4 = find_opening_closing(list_files_in_directory('new_split_txt/ln/2019/pp4-2019bt'))
print(tes, tes2, tes3, tes4)

tes5 = find_body_struktur(list_files_in_directory('new_split_txt/ln/2019/pp4-2019bt'))
print(tes5)


# +
def read_file(file_path):
    file_path = file_path.strip()
    if file_path == '':
        return
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except Exception as e:
        print(e)
        raise

def read_after_first_blank_line(file_content):
    content = []
    blank_line_found = False

    for line in file_content.splitlines():
        if line.strip() == "":
            blank_line_found = True
            continue

        if blank_line_found:
            content.append(line)

    return '\n'.join(content)


# -

def process_dataset(part, new_df):
    total_rows = len(new_df)
    part_size = total_rows

    for i in range(1):
        print(f'Start Creating Dataset {part}...')

        idx = new_df.iloc[i*part_size:(i+1)*part_size]
        dataset = Dataset.from_pandas(idx)

        print(f'Start Saving Dataset {part}...')
        print(f'Saving at ../dataset-surface-info/{part}/{part}')
        dataset.save_to_disk(f'../dataset-surface-info/{part}/{part}')


def process_dataset_split(part, new_df, split):
    print(f'Start Creating Dataset {part} {split}...')

    dataset = Dataset.from_pandas(new_df)

    print(f'Start Saving Dataset {part} {split}...')
    print(f'Saving at ../dataset-surface-info/{part}/{part}-{split}')
    dataset.save_to_disk(f'../dataset-surface-info/{part}/{part}-{split}')


def process_dataset_chunk(part, dataset, idx):
    print(f'Start Saving Dataset {part} {idx}...')
    print(f'Saving at ../dataset-surface-info/{part}/{part}-{idx}')
    dataset.save_to_disk(f'../dataset-surface-info/{part}/{part}-{idx}')


# +
def read_df_excluded():
    file_name = 'amandemen.csv'
    print(f'Start Reading Files {file_name}...')
    df = pd.read_csv(file_name)
    return df

def process_df_excluded(df, col_name):
    df['reg_id_lower'] = df[col_name].apply(lambda x: x.lower())
    df['reg_id'] = df[col_name]
    df = df[['reg_id', 'reg_id_lower']]
    return df


# +
def read_df():
    file_name = 'core/regulatory_map_surface_info.csv'
    print(f'Start Reading Files {file_name}...')
    df = pd.read_csv(file_name)
    return df

def process_df(df, excluded_df):
    df['label'] = df['regulatory'].apply(lambda x: x.split('_')[0])
    df['regulatory_lower'] = df['regulatory'].apply(lambda x: x.lower())
    
    value_counts = df['label'].value_counts()
#     labels_with_min_20_occurrences = value_counts[value_counts >= 20].index
#     df = df[df['label'].isin(labels_with_min_20_occurrences)]
    df = df[~df['regulatory_lower'].isin(excluded_df['reg_id_lower'])]
    
    df = df[['regulatory', 'label', 'file_txt', 'file_ttl']]
    return df


# +
def create_dataset_separate_surface(part, df):
    df = df.copy()
    
    if part == 'opening':
        idx = 3
    elif part == 'closing':
        idx = 4
    elif part == 'body struktur':
        idx = 5
    
    df['file_ttl'] = df['file_ttl'].apply(lambda x: x.replace('new_2_turtle_files', f'new_{idx}_turtle_files'))
    df['triples'] = df['file_ttl'].apply(lambda x: read_after_first_blank_line(read_file(x)))
    
    df['folder_txt'] = df['file_txt'].apply(lambda x: x.replace('new_1_text_files', 'new_split_txt').split('.')[0])
    df[['opening', 'opening_1', 'opening_2', 'closing']] = df['folder_txt'].apply(lambda x: pd.Series(find_opening_closing(list_files_in_directory(x))))
    
    if part == 'opening':
        df['txt'] = df['opening']
        df['txt_1'] = df['opening_1']
        df['txt_2'] = df['opening_2']
        
        df['file_txt'] = df['folder_txt'] + '/' + df['txt']
        df['file_txt_1'] = df['folder_txt'] + '/' + df['txt_1']
        df['file_txt_2'] = df['folder_txt'] + '/' + df['txt_2']
        
#         df['text'] = df['file_txt'].apply(lambda x: read_after_first_blank_line(read_file(x)))
#         df['text_1'] = df['file_txt_1'].apply(lambda x: read_after_first_blank_line(read_file(x)))
#         df['text_2'] = df['file_txt_2'].apply(lambda x: read_after_first_blank_line(read_file(x)))

        df['text'] = df['file_txt'].apply(lambda x: read_file(x).strip())
        df['text_1'] = df['file_txt_1'].apply(lambda x: read_file(x).strip())
        df['text_2'] = df['file_txt_2'].apply(lambda x: read_file(x).strip())
        
        new_df = df[['regulatory', 'label', 'text', 'text_1', 'text_2', 'triples']]
#         new_df = df[['regulatory', 'label', 'file_txt', 'file_txt_1', 'file_txt_2', 'triples']]
        
    elif part == 'closing':
        df['txt'] = df['closing']
        
        df['file_txt'] = df['folder_txt'] + '/' + df['txt']
        df['text'] = df['file_txt'].apply(lambda x: read_file(x).strip())

        new_df = df[['regulatory', 'label', 'text', 'triples']]
        
    elif part == 'body struktur':
        df['file_txt'] = df['folder_txt'] + '/' + '_.txt'
        df['text'] = df['file_txt'].apply(lambda x: read_file(x).strip())
        
        df['file_txt_1'] = df['folder_txt'] + '/' + '*.txt'
        df['text_1'] = df['file_txt_1'].apply(lambda x: read_file(x).strip())
        
        new_df = df[['regulatory', 'label', 'text', 'text_1', 'triples']]
        
    return new_df


# -

# ## Opening

df_excluded = read_df_excluded()
df_excluded = process_df_excluded(df_excluded, 'regulatory')
df_excluded

df = read_df()
df = process_df(df, df_excluded)
df

df['label'].value_counts()

new_df = create_dataset_separate_surface('opening', df)
new_df.reset_index(drop=True, inplace=True)
new_df

process_dataset('new-opening', new_df)

# +
# X = new_df.drop(columns=['label'])
# y = new_df['label']

# X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True, stratify=y)
# X_val, X_test, y_val, y_test = train_test_split(X_testval, y_testval, test_size=0.5, random_state=42, shuffle=True)

# +
# train = pd.concat([X_train, y_train], axis=1)
# train.reset_index(drop=True, inplace=True)

# test = pd.concat([X_test, y_test], axis=1)
# test.reset_index(drop=True, inplace=True)

# val = pd.concat([X_val, y_val], axis=1)
# val.reset_index(drop=True, inplace=True)

# +
# process_dataset_split('new-opening', train, 'train')
# process_dataset_split('new-opening', test, 'test')
# process_dataset_split('new-opening', val, 'val')
# -

# ## Closing

new_df_2 = create_dataset_separate_surface('closing', df)
new_df_2.reset_index(drop=True, inplace=True)
new_df_2

process_dataset('new-closing', new_df_2)

# +
# train_closing = new_df_2[new_df_2['regulatory'].isin(train['regulatory'])]
# train_closing.reset_index(drop=True, inplace=True)

# test_closing = new_df_2[new_df_2['regulatory'].isin(test['regulatory'])]
# test_closing.reset_index(drop=True, inplace=True)

# val_closing = new_df_2[new_df_2['regulatory'].isin(val['regulatory'])]
# val_closing.reset_index(drop=True, inplace=True)

# +
# process_dataset_split('closing', train_closing, 'train')
# process_dataset_split('closing', test_closing, 'test')
# process_dataset_split('closing', val_closing, 'val')
# -

# ## Body Struktur

new_df_3 = create_dataset_separate_surface('body struktur', df)
new_df_3.reset_index(drop=True, inplace=True)
new_df_3

process_dataset('new-body-struktur', new_df_3)



# ### Manual

df_split = pd.read_csv('core/train_test_split.csv')
df_split

train = df_split[df_split['type'] == 'train'][['regulatory']]
test = df_split[df_split['type'] == 'test'][['regulatory']]
val = df_split[df_split['type'] == 'val'][['regulatory']]

train_closing = new_df_2[new_df_2['regulatory'].isin(train['regulatory'])]
test_closing = new_df_2[new_df_2['regulatory'].isin(test['regulatory'])]
val_closing = new_df_2[new_df_2['regulatory'].isin(val['regulatory'])]

process_dataset_split('closing', train_closing, 'train')
process_dataset_split('closing', test_closing, 'test')
process_dataset_split('closing', val_closing, 'val')

# ## Info

train_closing_2 = train_closing.copy()
test_closing_2 = test_closing.copy()
val_closing_2 = val_closing.copy()

# +
train_closing_2['type'] = 'train'
test_closing_2['type'] = 'test'
val_closing_2['type'] = 'val'

splitted_df = pd.concat([train_closing_2, test_closing_2, val_closing_2], ignore_index=True)
splitted_df = splitted_df[['regulatory', 'type']]
splitted_df
# -

splitted_df.to_csv("core/train_test_split.csv", index=False)

# # Chunking

from collections import Counter

opening_dataset_name = "../dataset-surface-info/new-dataset/new-opening/new-opening"
opening_dataset = load_from_disk(opening_dataset_name)
closing_dataset_name = "../dataset-surface-info/new-dataset/new-closing/new-closing"
closing_dataset = load_from_disk(closing_dataset_name)
body_st_dataset_name = "../dataset-surface-info/new-body-struktur/new-body-struktur"
body_st_dataset = load_from_disk(body_st_dataset_name)


def split_stratify(dataset, stratify_by_column, init=None):
    dataset = dataset.sort('regulatory')
    if init == None:
        dct = dataset.train_test_split(test_size=0.5, seed=42, stratify_by_column=stratify_by_column)
    else:
        dct = dataset.class_encode_column(stratify_by_column).train_test_split(test_size=0.5, seed=42, stratify_by_column=stratify_by_column)
    train, test = dct['train'], dct['test']
    return train, test


def create_chunk(dataset, part, stratify_by=None):
    label_counts = Counter(dataset[stratify_by])
    labels_to_keep = [label for label, count in label_counts.items() if count >= 20]
    filtered_dataset = dataset.filter(lambda x: x[stratify_by] in labels_to_keep)
    c_filtered_dataset = dataset.filter(lambda x: x[stratify_by] not in labels_to_keep)

    a, b = split_stratify(filtered_dataset, stratify_by, True)
    a1, a2 = split_stratify(a, stratify_by)
    b1, b2 = split_stratify(b, stratify_by)
    a11, a12 = split_stratify(a1, stratify_by)
    a21, a22 = split_stratify(a2, stratify_by)
    b11, b12 = split_stratify(b1, stratify_by)
    b21, b22 = split_stratify(b2, stratify_by)

    lst = [a11, a12, a21, a22, b11, b12, b21, b22, c_filtered_dataset]

    for i in range(len(lst)):
        process_dataset_chunk(part, lst[i], i+1)


# +
def process_dataset_chunk(part, dataset, idx):
    print(f'Start Saving Dataset {part} {idx}...')
    print(f'Saving at ../dataset-surface-info/{part}/{part}-{idx}')
    dataset.save_to_disk(f'../dataset-surface-info/{part}/{part}-{idx}')

# def create_chunk(dataset, part, stratify_by=None):
#     label_counts = Counter(dataset[stratify_by])
#     labels_to_keep = [label for label, count in label_counts.items() if count == 14]
#     filtered_dataset = dataset.filter(lambda x: x[stratify_by] in labels_to_keep)

#     a, b = split_stratify(filtered_dataset, stratify_by, True)
#     a1, a2 = split_stratify(a, stratify_by)
#     b1, b2 = split_stratify(b, stratify_by)
#     a11, a12 = split_stratify(a1, stratify_by)
#     a21, a22 = split_stratify(a2, stratify_by)
#     b11, b12 = split_stratify(b1, stratify_by)
#     b21, b22 = split_stratify(b2, stratify_by)

#     lst = [a11, a12, a21, a22, b11, b12, b21, b22]

#     for i in range(len(lst)):
#         process_dataset_chunk(part, lst[i], i+1)


# -

create_chunk(opening_dataset, 'new-new-opening', stratify_by='label')

create_chunk(closing_dataset, 'new-new-closing', stratify_by='label')

create_chunk(body_st_dataset, 'new-new-body-struktur', stratify_by='label')

# # Cek

opening_dataset_name_1 = "../dataset-surface-info/opening/opening-val"
opening_dataset_1 = load_from_disk(opening_dataset_name_1)
closing_dataset_name_1 = "../dataset-surface-info/new-opening/new-opening-val"
closing_dataset_1 = load_from_disk(closing_dataset_name_1)

opening_dataset_name_1 = "../dataset-surface-info/new-dataset/new-opening/new-opening"
opening_dataset_1 = load_from_disk(opening_dataset_name_1)
closing_dataset_name_1 = "../dataset-surface-info/new-dataset/new-closing/new-closing"
closing_dataset_1 = load_from_disk(closing_dataset_name_1)
body_st_dataset_name_1 = "../dataset-surface-info/new-body-struktur/new-body-struktur"
body_st_dataset_1 = load_from_disk(body_st_dataset_name_1)

opening_dataset_1[:5]

closing_dataset_1[:5]

body_st_dataset_1[:5]



test_opening = load_from_disk('../dataset-surface-info/new-new-opening/new-new-opening-1')
test_closing = load_from_disk('../dataset-surface-info/new-new-closing/new-new-closing-1')
test_body = load_from_disk('../dataset-surface-info/new-new-body-struktur/new-new-body-struktur-1')



# # Concat 2 Dataframe

# +
from datasets import load_from_disk, concatenate_datasets, ClassLabel


def concat_dataset(part, idx, data_1, data_2):
    class_names1 = data_1.features['label'].names
    class_names2 = data_2.features['label'].names

    all_class_names = sorted(set(class_names1 + class_names2))

    new_features = data_1.features.copy()
    new_features['label'] = ClassLabel(names=all_class_names)

    def update_labels(example):
        example['label'] = all_class_names.index(data_1.features['label'].names[example['label']])
        return example

    updated_data_1 = data_1.map(update_labels, features=new_features)

    def update_labels2(example):
        example['label'] = all_class_names.index(data_2.features['label'].names[example['label']])
        return example

    updated_data_2 = data_2.map(update_labels2, features=new_features)

    concatenated_dataset = concatenate_datasets([updated_data_1, updated_data_2])
    concatenated_dataset.save_to_disk(f'../dataset-surface-info/{part}/{part}-{idx}')


# +
parts = ['new-closing', 'new-opening', 'new-body-struktur']

for part in parts:
    for idx in range(1, 9):
        data_1 = load_from_disk(f'../dataset-surface-info/new-dataset/{part}/{part}-{idx}')
        data_2 = load_from_disk(f'../dataset-surface-info/new-{part}/new-{part}-{idx}')
        concat_dataset(part, idx, data_1, data_2)
# -

data = load_from_disk(f'../dataset-surface-info/new-dataset/new-closing/new-closing-1')
data['regulatory'][-1]

# ## Surface

# +
# start = time.time()
# print('Start')

# +
# print('Start Reading Files...')
# df = pd.read_csv('core/regulatory_map_surface_info.csv')
# df['text'] = df['file_txt'].apply(read_file)
# df['triples'] = df['file_ttl'].apply(read_file)
# df['label'] = df['regulatory'].apply(lambda x: x.split('_')[0])
# new_df = df[['regulatory', 'label', 'text', 'triples']]

# +
# value_counts = new_df['label'].value_counts()
# labels_with_min_4_occurrences = value_counts[value_counts >= 4].index
# new_df = new_df[new_df['label'].isin(labels_with_min_4_occurrences)]

# +
## Surface

# def process_dataset():
#     total_rows = len(new_df)
#     part_size = total_rows

#     for i in range(3):
#         print(f'Start Creating Dataset {i+1}...')

#         idx = new_df.iloc[i*part_size:(i+1)*part_size]
#         dataset = Dataset.from_pandas(idx)

#         print(f'Start Saving Dataset {i+1}...')
#         dataset.save_to_disk(f'../dataset-surface-info/dataset-{i+1}')
    
# process_dataset()
# -

# ## All

# +
# import cupy as cp
# from datasets import Dataset

# def process_and_save_dataset():
#     total_rows = len(new_df)
#     part_size = total_rows // 5

#     for i in range(5):
#         print(f'Start Creating Dataset {i+1}...')

#         idx = new_df.iloc[i*part_size:(i+1)*part_size]
        
#         regulatory_gpu = cp.asarray(idx['regulatory'])
#         text_gpu = cp.asarray(idx['text'])
#         triples_gpu = cp.asarray(idx['triples'])

#         regulatory_cpu = cp.asnumpy(regulatory_gpu)
#         text_cpu = cp.asnumpy(text_gpu)
#         triples_cpu = cp.asnumpy(triples_gpu)

#         dataset = Dataset.from_dict({"regulatory": regulatory_cpu, 
#                                       "text": text_cpu, 
#                                       "triples": triples_cpu})

#         print(f'Start Saving Dataset {i+1}...')
#         dataset.save_to_disk(f'../dataset/dataset-{i+1}')
    
# process_and_save_dataset()

# +
# print('Start Creating Dataset...')

# total_rows = len(new_df)
# part_size = total_rows // 10

# for i in range(5):
#     print(f'Start Creating Dataset {i+1}...')
    
#     idx = new_df.iloc[i*part_size:(i+1)*part_size]
#     dataset = Dataset.from_pandas(idx)
    
#     print(f'Start Saving Dataset {i+1}...')
#     dataset.save_to_disk(f'../dataset/dataset-{i+1}')

# # 1st
# i = 1
# print(f'Index: {0*part_size}:{1*part_size}')
# idx = new_df.iloc[:part_size]
# dataset = Dataset.from_pandas(idx)

# # 2nd
# i = 2
# print(f'Index: {1*part_size}:{2*part_size}')
# idx = new_df.iloc[part_size:2*part_size]
# dataset = Dataset.from_pandas(idx)

# # 3rd
# i = 3
# print(f'Index: {2*part_size}:{3*part_size}')
# idx = new_df.iloc[2*part_size:3*part_size]
# dataset = Dataset.from_pandas(idx)

# # 4th
# i = 4
# print(f'Index: {3*part_size}:{4*part_size}')
# idx = new_df.iloc[3*part_size:4*part_size]
# dataset = Dataset.from_pandas(idx)

# # 5th
# i = 5
# print(f'Index: {4*part_size}:{5*part_size}')
# idx = new_df.iloc[4*part_size:5*part_size]
# dataset = Dataset.from_pandas(idx)

# # 6th
# i = 6
# print(f'Index: {5*part_size}:{6*part_size}')
# idx = new_df.iloc[5*part_size:6*part_size]
# dataset = Dataset.from_pandas(idx)

# # 7th
# i = 7
# print(f'Index: {6*part_size}:{7*part_size}')
# idx = new_df.iloc[6*part_size:7*part_size]
# dataset = Dataset.from_pandas(idx)

# # 8th
# i = 8
# print(f'Index: {7*part_size}:{8*part_size}')
# idx = new_df.iloc[7*part_size:8*part_size]
# dataset = Dataset.from_pandas(idx)

# # 9th
# i = 9
# print(f'Index: {8*part_size}:{9*part_size}')
# idx = new_df.iloc[8*part_size:9*part_size]
# dataset = Dataset.from_pandas(idx)

# # 10th
# i = 10
# print(f'Index: {9*part_size}:{10*part_size}')
# idx = new_df.iloc[9*part_size:10*part_size]
# dataset = Dataset.from_pandas(idx)

# +
# print(f'Start Saving Dataset {i}...')
# dataset.save_to_disk(f'../dataset/dataset_{i}')

# +
# print('End')
# end = time.time()
# print('Time:', end - start)
