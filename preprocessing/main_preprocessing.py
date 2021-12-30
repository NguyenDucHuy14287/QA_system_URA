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
    os.makedirs(f"{cleaned_data_path}/qa", exist_ok=True)
    # dataset = load_dataset('narrativeqa', split=f'{set_name}[:1%]')
    dataset = load_dataset('narrativeqa', split=f'{set_name}')
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]
        # doc
        if doc["id"] not in doc_id_set:
            doc_id_set.add(doc["id"])
            print(f"Preprocessing doc id {doc['id']}")
            clean_doc = preprocess_doc(doc["text"], doc["start"], doc["end"])
            for part in clean_doc:
                if part.strip() != "":
                    append_text(part.strip(), f"{cleaned_data_path}/{doc['id']}.txt")

        # qa
        qa_lst = []
        qa_lst.append(text["question"]["text"].lower())
        for ans in text["answers"]:
            qa_lst.append(ans["text"].lower())
        append_text("[SEP]".join(qa_lst), f"{cleaned_data_path}/qa/{doc['id']}.txt")

if __name__ == '__main__':
    use_bert_pretrained_tokenize = True

    ### Clean dataset
    dataset_list = ["train", "test", "validation"]
    for name in dataset_list:
        cleaned_data_path = f"data/preprocessed/{name}"
        if not os.path.isdir(cleaned_data_path):
            clean_data(name, cleaned_data_path)


    ### Create tokenize
    if use_bert_pretrained_tokenize:
        tokenizer = hf_tokenizer(from_bert_pretrained=True)
    else:
        # Train tokenizer from corpus.
        tokenizer = hf_tokenizer(from_bert_pretrained=False)
        tokenizer_path_file = "data/tokenizer_vocab.txt"
        if os.path.isfile(tokenizer_path_file):
            tokenizer.load_tokenizer(tokenizer_path_file)
        else:
            for name in dataset_list:
                tokenizer.train_tokenizer(f"data/preprocessed/{name}")
                tokenizer.train_tokenizer(f"data/preprocessed/{name}/qa")
            tokenizer.save_tokenizer(tokenizer_path_file)


    # ### Load dataset from file (each dataset is a lst of document, each document is a lst of paragraphs)
    train_set = load_dataset_from_path("data/preprocessed/train")           # 1099
    test_set = load_dataset_from_path("data/preprocessed/test")             # 354
    valid_set = load_dataset_from_path("data/preprocessed/validation")      # 115


    ### Tokenize first doc
    first_doc = train_set[0]
    print(first_doc["id"])

    # token full text
    doc_text_token = tokenizer.encode_sentence(first_doc["clean_text"])
    print(doc_text_token.tokens)
    print(doc_text_token.ids)

    # token paragraph list
    doc_paragraph_token = tokenizer.encode_sentence_lst(first_doc["paragraph"])
    print(doc_paragraph_token)

    # token qa list
    qa_token = [tokenizer.encode_sentence_lst(qa_tup) for qa_tup in first_doc["qa"]]
    print(qa_token)





