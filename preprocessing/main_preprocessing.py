import os
import re
from datasets import load_dataset

from preprocess_utils import *
from tokenize_utils import *

def check_doc_by_id():
    # dict_keys(['id', 'kind', 'url', 'file_size', 'word_count', 'start', 'end', 'summary', 'text'])
    # dataset = load_dataset('narrativeqa', split='validation', keep_in_memory=True, save_infos=True)
    dataset = load_dataset('narrativeqa', split='train[:10%]')
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]
        if doc["id"] in doc_id_set:
            continue
        else:
            doc_id_set.add(doc["id"])
            if doc["id"] == "0fec426130719e5a0ede16a2944f437415917462":
                pass

def clean_data(set_name, cleaned_data_path):
    print(f"=== [INFO] Cleaning {set_name} set and save to \'{cleaned_data_path}\'")
    os.makedirs(cleaned_data_path, exist_ok=True)
    # dataset = load_dataset('narrativeqa', split=f'{set_name}[:10%]')
    dataset = load_dataset('narrativeqa', split=f'{set_name}')
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]
        if doc["id"] in doc_id_set or not is_english(doc["text"]):
            continue
        else:
            doc_id_set.add(doc["id"])
            print(f"Preprocessing doc id {doc['id']}")
            clean_doc = preprocess(doc["text"], doc["start"], doc["end"])
            for part in clean_doc:
                append_text(part, f"{cleaned_data_path}/{doc['id']}.txt")

if __name__ == '__main__':
    use_bert_pretrained = False

    ### Clean dataset
    dataset_list = ["train", "test", "validation"]
    for name in dataset_list:
        cleaned_data_path = f"data/preprocessed/{name}"
        if not os.path.isdir(cleaned_data_path):
            clean_data(name, cleaned_data_path)


    ### Create tokenize
    if use_bert_pretrained:
        tokenizer = hf_tokenizer(from_bert_pretrained=True)
    else:
        # Train tokenizer from corpus.
        tokenizer = hf_tokenizer(from_bert_pretrained=False)
        tokenizer_path_file = "data/tokenizer_vocab.txt"
        if os.path.isfile(tokenizer_path_file):
            tokenizer.load_tokenizer(tokenizer_path_file)
        else:
            tokenizer.train_tokenizer("data/preprocessed/train")
            tokenizer.train_tokenizer("data/preprocessed/test")
            tokenizer.train_tokenizer("data/preprocessed/validation")
            tokenizer.save_tokenizer(tokenizer_path_file)


    ### Load dataset from file (each dataset is a lst of document, each document is a lst of paragraphs)
    train_set = load_dataset_from_path("data/preprocessed/train")
    test_set = load_dataset_from_path("data/preprocessed/test")
    valid_set = load_dataset_from_path("data/preprocessed/validation")


    ### Tokenize dataset
    train_tokens = [tokenizer.encode_sentence_lst(doc) for doc in train_set]
    test_tokens = [tokenizer.encode_sentence_lst(doc) for doc in test_set]
    valid_tokens = [tokenizer.encode_sentence_lst(doc) for doc in valid_set]










