import os
import re
import random
import logging
import spacy
from spacy import displacy
from langdetect import detect
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, set_caching_enabled
import unidecode
import time
from utils import append_text, write_pickle

logging.basicConfig(level=logging.INFO)

url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
rm_lst = [url_regex, "\t"]

start_token_extra_lst = ["produced by", "start of the project", "start of this project", "written by"]
end_token_extra_lst = ["start full license", "end of the project", "end of project"]

count = 0


def is_html(text, thresh_hold=2):
    html_tag = BeautifulSoup(text, "html.parser").find()
    if html_tag is None:
        return False
    elif len(html_tag) <= thresh_hold:
        return False
    else:
        return True


def remove_accent(text):
    return unidecode.unidecode(text)


def slice_content(raw_text, start_token, end_token):
    start_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in start_token_extra_lst + [start_token]]
    end_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in end_token_extra_lst + [end_token]]

    start_pos_lst = [raw_text.lower().find(i) for i in start_token_lst]
    end_pos_lst = [raw_text.lower().rfind(i) for i in end_token_lst]

    min_start_pos = max(start_pos_lst)
    min_end_pos = len(raw_text)
    end_token1 = ""

    for start_pos in start_pos_lst:
        if start_pos != -1 and start_pos <= min_start_pos:
            min_start_pos = start_pos

    if min_start_pos == -1:
        min_start_pos = 0

    for index, end_pos in enumerate(end_pos_lst):
        if end_pos != -1 and end_pos <= min_end_pos:
            min_end_pos = end_pos
            end_token1 = end_token_lst[index]

    return raw_text[min_start_pos:min_end_pos + len(end_token1)]


def re_sub_text(text):
    # text = re.sub("'ll", " will", text)
    # text = re.sub("n't", " not", text)

    # remove special str
    for pattern in rm_lst:
        text = re.sub(pattern, "", text)

    # replace ?, ! by .
    text = re.sub(r'[!?]', '.', text).strip()

    # remove other punc and lower text
    # text = re.sub(r'[^a-zA-Z0-9\s.,()!?&\']', ' ', text.lower()).strip()
    text = re.sub(r'[^a-zA-Z\s.\']', '', text).strip()

    return text


def clean_sentence(sen):
    token = " [SEP] "
    sen = sen.replace(".", token)
    if not sen.strip().endswith(token):
        sen = sen + token
    sen = re.sub(r"\s+", " ", sen)
    return sen.lower().strip()


def preprocess_qa(sen):
    sen = sen.replace('?', '').replace('.', '') + " [SEP] "
    return sen.lower()


def split_and_concat_paragpraph(sliced_text):
    split_lst = sliced_text.split("\n")
    return_lst = []
    text = ""
    for sent in split_lst:
        if sent == "":
            if text != "":
                cleaned_text = clean_sentence(text)
                return_lst.append(cleaned_text)
                text = ""
        else:
            text = text + " " + sent

    if text.strip() != "":
        cleaned_text = clean_sentence(text)
        return_lst.append(cleaned_text)

    return return_lst


def preprocess_doc(raw_text, start_token, stop_token):
    if is_html(raw_text):
        scrtext_html = BeautifulSoup(raw_text, features="lxml").body.find('td', attrs={'class': 'scrtext'})
        if scrtext_html is not None:
            raw_text = scrtext_html.text
        else:
            raw_text = BeautifulSoup(raw_text, "lxml").text
    preprocessed_text = re_sub_text(raw_text)
    sliced_text = slice_content(preprocessed_text, start_token, stop_token)
    paragraphs = split_and_concat_paragpraph(sliced_text)
    return paragraphs


def clean_data(dataset_name, raw_data_path, cleaned_data_path):
    global count
    logging.info(f"=== Cleaning {dataset_name} set and save to \'{cleaned_data_path}\'")

    data_files = [f'{raw_data_path}/{dataset_name}_{i}.json' for i in range(3)]

    dataset = load_dataset('json', data_files=data_files)
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]

        # doc
        if doc["id"] != "3ee65995071a0e70027e74a9b7735a734ba43bc7":  # french
            if doc["id"] not in doc_id_set:
                count += 1
                print("====== Processing doc id:", doc["id"], doc["kind"], count)
                doc_id_set.add(doc["id"])

                if doc["id"] == "37fa67ed55fc62766b9a5f0edcafcc360131aebb":  # unicode format failed
                    correct_text = "".join([remove_accent(char) for char in doc["text"]])
                    correct_text = re.sub("AC/AA", "", correct_text)
                    correct_text = re.sub('i>>\?A-A>>A\?', "", correct_text)
                    correct_text = re.sub("_", "", correct_text)
                    clean_doc = preprocess_doc(correct_text, doc["start"], doc["end"])
                else:
                    clean_doc = preprocess_doc(doc["text"], doc["start"], doc["end"])

                for paragraph in clean_doc:
                    if paragraph != "":
                        append_text(paragraph, cleaned_data_path, f"{doc['id']}.txt")

            # qa
            # print("=== Processing qa:", doc["id"], doc["kind"])
            qa_lst = []
            processed_q = preprocess_qa(text["question"]["text"])
            qa_lst.append(processed_q.replace('?', ''))
            for ans in text["answers"]:
                processed_a = preprocess_qa(ans["text"])
                qa_lst.append(processed_a)
            append_text("".join(qa_lst), f"{cleaned_data_path}/qa", f"{doc['id']}.txt")


if __name__ == '__main__':
    ### Clean dataset
    dataset_list = ["train", "test", "validation"]
    for dataset_name in dataset_list:
        count = 0
        raw_data_path = f"./data/raw"
        cleaned_data_path = f"./data/cleaned/{dataset_name}"
        if not os.path.isdir(cleaned_data_path):
            clean_data(dataset_name, raw_data_path, cleaned_data_path)




