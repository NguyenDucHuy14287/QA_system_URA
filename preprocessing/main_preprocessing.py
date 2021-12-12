import os
import re
from datasets import load_dataset

from utils import *

def check_doc_by_id():
    dataset = load_dataset('narrativeqa', split='train[:10%]')
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]
        if doc["id"] in doc_id_set:
            continue
        else:
            doc_id_set.add(doc["id"])
            if doc["id"] == "doc_id":
                pass

def preprocess_data():
    dataset = load_dataset('narrativeqa', split='train[:10%]')
    doc_id_set = set()
    paragpraph_lst = []
    for text in dataset:
        doc = text["document"]
        if doc["id"] in doc_id_set:
            continue
        else:
            doc_id_set.add(doc["id"])
            if is_html(doc["text"]):
                append_text(doc["id"], "html_doc_id.txt")
            elif not is_english(doc["text"]):
                append_text(doc["id"], "not_english_doc_id.txt")
            else:
                preprocess_text = preprocessing_text(doc["text"])
                slice_text = slice_content(preprocess_text, doc["start"], doc["end"])
                paragpraph = split_and_concat_paragpraph(slice_text)
                # write_text(doc["text"], "raw.txt")
                # write_text(preprocess_text, "preprocessing.txt")
                paragpraph_lst.append(paragpraph)

    print(paragpraph_lst)

if __name__ == '__main__':
    # check_doc_by_id()
    preprocess_data()






