import chardet
# after preprocessing the files saved from this script, we get 20 users who meet the criteria of filtering for both test and train

def check_encoding(input_file):
    with open(input_file, 'rb') as file:
        result = chardet.detect(file.read())
    charenc = result["encoding"]
    print(charenc)

def split_file(input_file):
    with open(input_file, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
    
    # Save the first 1000 lines to 'first_1000.txt'
    with open('./data/tsmc2014/dataset_TSMC2014_TKY.txt', 'w') as file1:
        file1.writelines(lines[:153750])

# Usage
# check_encoding('./data/tsmc2014/original_dataset_TSMC2014_TKY.txt')
split_file('./data/tsmc2014/original_dataset_TSMC2014_TKY.txt')
