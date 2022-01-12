import os
import json
import pickle
from pathlib import Path


def write_text(text, folder, path="data_text.txt"):
    Path(folder).mkdir(parents=True, exist_ok=True)
    f = open(path, "w", encoding="utf-8")
    f.write(text)
    f.close()

def append_text(text, folder, file_name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    f = open(f"{folder}/{file_name}", 'a', encoding="utf-8")
    f.write(text + "\n")
    f.close()

def read_text(folder, path):
    f = open(path, "r", encoding="utf-8")
    text = f.read()
    f.close()
    return text

def write_json(data, folder, file_name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    f = open(f"{folder}/{file_name}", "w", encoding="utf-8")
    json.dump(data, f)

def write_pickle(object, folder, file_name):
    Path(folder).mkdir(parents=True, exist_ok=True)
    with open(f"{folder}/{file_name}", "wb") as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def load_dataset_from_path(data_path):
    file_lst = [file_name for file_name in os.listdir(data_path) if file_name[-4:] == ".txt"]
    data_set = []
    for file_name in file_lst:
        doc = {"id": file_name[:-4], "clean_text": None, "paragraph": None, "qa": []}

        full_text = read_text(f"{data_path}/{file_name}")
        doc["clean_text"] = full_text

        paragraph = full_text.split("\n")
        doc["paragraph"] = paragraph[:-1]

        full_qa = read_text(f"{data_path}/qa/{file_name}")
        pa_lst = full_qa.split("\n")
        for qa in pa_lst[:-1]:
            qa_lst = qa.split("[SEP]")
            for ans in qa_lst[1:]:
                doc["qa"].append((qa_lst[0], ans))

        data_set.append(doc)

    return data_set

