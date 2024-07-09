import os
import re


for root, dirs, _ in os.walk('./dataset-final/new_1_text_files'):
    for directory in dirs:
        f1 = os.path.join(root, directory)
        f2 = f1.replace('./dataset-final/new_1_text_files', './dataset-final/new_split_txt', 1)
        if not os.path.exists(f2):
            os.makedirs(f2)


list_files = []
for root, dirs, files in os.walk('./dataset-final/new_1_text_files'):
    for file in files:
        if file.endswith(".txt"):
            list_files.append(os.path.join(root, file))

# +
def split_text(file_path):
    os.makedirs(f"./dataset-final/split_txt/{file_path[33:-4]}")
    
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()

    curr = ''
    idx = 0
    for line in lines:
#         if 'P E N J E L A S A N' in line:
#             print([line.strip()])
        pattern1 = r'^\s*Pasal\s+\d+[A-Z]?\s*$'
        pattern2 = r'^(\s*p\s*e\s*n\s*j\s*e\s*l\s*a\s*s\s*a\s*n\s*|\s*l\s*a\s*m\s*p\s*i\s*r\s*a\s*n\s*(\s+[IVXLCDM\d]+)?)\s*$'
        
        if re.match(pattern1, line):
            with open(f"./dataset-final/split_txt/{file_path[33:-4]}/{idx}.txt", "w", encoding="utf8") as file:
                file.write(curr)
#             print(curr)
            curr = line + '\n'
            idx += 1
        elif re.match(pattern2, line, re.IGNORECASE) and stop:
            with open(f"./dataset-final/split_txt/{file_path[33:-4]}/{idx}.txt", "w", encoding="utf8") as file:
                file.write(curr)
#             print(curr)
            return
        else:
            curr += line + '\n'
        
    with open(f"./dataset-final/split_txt/{file_path[33:-4]}/{idx}.txt", "w", encoding="utf8") as file:
        file.write(curr)
        
#     print(curr)

# +
# split_text('./dataset-final/new_1_text_files/bn/2008/bn114-2008.txt')
# -

# Pls uncomment the code below

# +
# for file in list_files:
#     print(file)
#     # try:
#     split_text(file)
# #     break


# +
# list_files[:1]

# +
# import re

# # Pola regex
# pattern = r'^(\s*p\s*e\s*n\s*j\s*e\s*l\s*a\s*s\s*a\s*n\s*|\s*l\s*a\s*m\s*p\s*i\s*r\s*a\s*n\s*(\s+[IVXLCDM\d]+)?)\s*$'

# # Contoh input
# input_string = "LAMPIRAN I"

# # Cek apakah input sesuai dengan pola
# if re.match(pattern, input_string, re.IGNORECASE):
#     print("Cocok!")
# else:
#     print("Tidak cocok!")


# -

# # Revisi

def split_text_0(file_path):
    os.makedirs(f"./dataset-final/new_split_txt/{file_path[33:-4]}")
    
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()

    curr = ''
    idx = 0
    stop = 0 # 1 if Diundangkan di is found
    for line in lines:
#         if 'P E N J E L A S A N' in line:
#             print([line.strip()])
        pattern1 = r'^\s*Pasal\s+\d+[A-Z]?\s*$'
        pattern2 = r'^(\s*p\s*e\s*n\s*j\s*e\s*l\s*a\s*s\s*a\s*n\s*|\s*l\s*a\s*m\s*p\s*i\s*r\s*a\s*n\s*(\s+[IVXLCDM\d]+)?).*$'
        pattern3 = r'^.*(\s*D\s*i\s*u\s*n\s*d\s*a\s*n\s*g\s*k\s*a\s*n).*$'
        pattern4 = r'^.*(\s*w\s*w\s*w\s*.\s*).*$'
        
        if re.match(pattern1, line):
            with open(f"./dataset-final/new_split_txt/{file_path[33:-4]}/{idx}.txt", "w", encoding="utf8") as file:
                file.write(curr.strip())
            curr = line + '\n'
            idx += 1
        elif re.match(pattern2, line, re.IGNORECASE) and stop:
            break
        elif re.match(pattern3, line):
            curr += line + '\n'
            stop = 1
        elif re.match(pattern4, line) and stop:
            break
        else:
            curr += line + '\n'
        
    with open(f"./dataset-final/new_split_txt/{file_path[33:-4]}/{idx}.txt", "w", encoding="utf8") as file:
        file.write(curr.strip())


# +
# for file in list_files:
#     print(file)
#     split_text_0(file)
# -

# ## Split Body

def split_text_2(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()
        
    curr = ''
    idx = 0 # 1 if found, 0 else
    mch = 0
    stop = 0
    for line in lines:
        pattern5 = r'^(\s*p\s*e\s*n\s*j\s*e\s*l\s*a\s*s\s*a\s*n\s*|\s*l\s*a\s*m\s*p\s*i\s*r\s*a\s*n\s*(\s+[IVXLCDM\d]+)?).*$'
        match5 = re.match(pattern5, line, re.IGNORECASE)
        if match5 and stop:
            break
        
        pattern1 = r'^\s*(Pasal\s*\d+)[A-Z]?\s*$'
        pattern2 = r'^\s*(Paragraf\s*\d+)[A-Z]?\s*$'
        pattern3 = r'^\s*(Bagian\s*[A-Z]{1}[a-z]+)\s*$'
        pattern4 = r'^\s*(BAB\s*[IVXLCDM\d]+)\s*$'
        pattern5 = r'^\s*(ttd).*$'
        
        match1 = re.match(pattern1, line)
        match2 = re.match(pattern2, line)
        match3 = re.match(pattern3, line)
        match4 = re.match(pattern4, line)
        match5 = re.match(pattern5, line, re.IGNORECASE)

        if match1:
            curr += match1.group(1) + '\n'
            mch = 1
            idx = 1
        elif match2:
            curr += match2.group(1) + '\n'
            mch = 2
            idx = 1
        elif match3:
            curr += match3.group(1) + '\n'
            mch = 3
            idx = 1
        elif match4:
            curr += match4.group(1) + '\n'
            mch = 4
            idx = 1
        elif match5:
            stop = 1
        else:
            if idx:
                if re.match(r'[A-Za-z0-9]+', line) and mch != 1:
                    curr += line + '\n'
                else:
                    idx = 0
                    mch = 0
                    
    with open(f"./dataset-final/new_split_txt/{file_path[33:-4]}/_.txt", "w", encoding="utf8") as file:
        file.write(curr)


def split_text_2_1(file_path):
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()
        
    curr = ''
    idx = 0 # 1 if found, 0 else
    mch = 0
    stop = 0
    for line in lines:
        pattern5 = r'^(\s*p\s*e\s*n\s*j\s*e\s*l\s*a\s*s\s*a\s*n\s*|\s*l\s*a\s*m\s*p\s*i\s*r\s*a\s*n\s*(\s+[IVXLCDM\d]+)?).*$'
        match5 = re.match(pattern5, line, re.IGNORECASE)
        if match5 and stop:
            break
        
        pattern1 = r'^\s*(Pasal\s*\d+)[A-Z]?\s*$'
        pattern2 = r'^\s*(Paragraf\s*\d+)[A-Z]?\s*$'
        pattern3 = r'^\s*(Bagian\s*[A-Z]{1}[a-z]+)\s*$'
        pattern4 = r'^\s*(BAB\s*[IVXLCDM\d]+)\s*$'
        pattern5 = r'^\s*(ttd).*$'
        
        match1 = re.match(pattern1, line)
        match2 = re.match(pattern2, line)
        match3 = re.match(pattern3, line)
        match4 = re.match(pattern4, line)
        match5 = re.match(pattern5, line, re.IGNORECASE)

        if match1:
            curr += match1.group(1) + '\n'
            mch = 1
            idx = 1
        elif match2:
            curr += match2.group(1) + '\n'
            mch = 2
            idx = 1
        elif match3:
            curr += match3.group(1) + '\n'
            mch = 3
            idx = 1
        elif match4:
            curr += match4.group(1) + '\n'
            mch = 4
            idx = 1
        elif match5:
            stop = 1
        else:
            if idx:
                if re.match(r'[A-Za-z0-9]+', line) and mch != 1:
                    pass
                else:
                    idx = 0
                    mch = 0
                    
    with open(f"./dataset-final/new_split_txt/{file_path[33:-4]}/*.txt", "w", encoding="utf8") as file:
        file.write(curr)



for file in list_files:
    print(file)
    split_text_2_1(file)

# +
# p1 = 'dataset-final/new_1_text_files/ln/2010/pp22-2010.txt'

# split_text_2_1(p1)

# +
# p = 'dataset-final/new_1_text_files/bn/2019/bn_1299-2019.txt'

# split_text_2(p)
# -

for file in list_files:
    print(file)
    split_text_2(file)


# # Split Opening

def split_text_3(file_path):  
    file_path = file_path.replace('new_1_text_files', 'new_split_txt')
    file_path = file_path.replace(file_path[-4:], '/0.txt')
    
    with open(file_path, "r", encoding="utf8") as file:
        lines = file.readlines()

    curr = ''
    curr1 = ''
    idx = 0  # 1 if menimbang is found
    cont = 0 # 1 if mengingat is found
    
    for line in lines:
        pattern1 = r'^.*M\s*e\s*n\s*i\s*m\s*b\s*a\s*n\s*g.*$'
        pattern2 = r'^.*M\s*e\s*n\s*g\s*i\s*n\s*g\s*a\s*t.*$'
        pattern3 = r'^.*(M\s*E\s*M\s*U\s*T\s*U\s*S\s*K\s*A\s*N|M\s*E\s*M\s*U\s*T\s*U\s*S\s*A\s*N).*$'
        pattern4 = r'^\s*(Pasal\s*\d+)[A-Z]?\s*$'
        
        if re.match(pattern1, line):
            curr1 += line + '\n'
            idx = 1
        elif re.match(pattern3, line):
            curr1 += line + '\n'
            cont = 0
        elif (re.match(pattern2, line) and idx) or cont:
            curr += line + '\n'
            cont = 1
        elif re.match(pattern4, line):
            break
        else:
            curr1 += line + '\n'
            
    curr = curr.strip()
    curr1 = curr1.strip()
            
    with open(f"./dataset-final/new_split_txt/{file_path[30:-6]}/-1.txt", "w", encoding="utf8") as file:
        file.write(curr)
        
    with open(f"./dataset-final/new_split_txt/{file_path[30:-6]}/-2.txt", "w", encoding="utf8") as file:
        file.write(curr1)

# +
# for file in list_files:
#     print(file)
#     split_text_3(file)

# +
# split_text_3('./dataset-final/new_1_text_files/perda/2010/perda_provinsi_banten_nomor_6_tahun_2010_11e58435326e3e089023313031393331.txt')
