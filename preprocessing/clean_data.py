import os
import re
import random
import logging
import spacy
from spacy import displacy
from langdetect import detect
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset, set_caching_enabled

from utils import append_text, write_pickle


logging.basicConfig(level=logging.INFO)
# set_caching_enabled(False)

#WRONG FONT
#f68f34f3a499dd2616049be9fb076a12e33f61b5


sequences_distribution = [i for i in range(1, 50000)]
entity_vocab = {}
entity_vocab_per_doc = {}
NER = spacy.load("en_core_web_sm")

NOT_ALLOWED_ENITITY_TYPE = ["DATE", "TIME", "CARDINAL"]

url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
rm_lst = [url_regex, "\t", "'s"]

start_token_extra_lst = ["produced by", "start of the project", "start of this project", "written by"]
end_token_extra_lst = ["start full license", "end of the project", "end of project"]


def is_html(text, thresh_hold=2):
    html_tag = BeautifulSoup(text, "html.parser").find()
    if html_tag is None:
        return False
    elif len(html_tag) <= thresh_hold:
        return False
    else:
        return True


def find_url(text):
    url = re.findall(url_regex, text)
    return [x[0] for x in url]


def mask_entity(text):
    global entity_vocab_per_doc
    global sequences_distribution

    entities = NER(text)
    for word in entities.ents:
        if re.search(r"[A-Z][a-z]+", word.text) and word.label_ not in NOT_ALLOWED_ENITITY_TYPE:
            if word.text not in entity_vocab_per_doc:
                number = random.choice(sequences_distribution)
                sequences_distribution.remove(number)
                mask = f"[ENT{number}]"
                entity_vocab_per_doc[word.text] = mask
            else:
                mask = entity_vocab_per_doc[word.text]
            text = re.sub(word.text, mask, text)
    return text


def slice_content(raw_text, start_token, end_token):
    start_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in start_token_extra_lst + [start_token]]
    end_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in end_token_extra_lst + [end_token]]

    start_pos_lst = [raw_text.lower().find(i) for i in start_token_lst]
    end_pos_lst = [raw_text.lower().rfind(i) for i in end_token_lst]
    max_start_pos = 0
    min_end_pos = len(raw_text)
    end_token1 = ""

    #choose token 'produced by' or 'written by'
    if  start_pos_lst[0] > -1 and start_pos_lst[3] > -1:
        if start_pos_lst[0] > start_pos_lst[3]:
            del start_pos_lst[0]
        else:
            del start_pos_lst[3]
    max_start_pos = max(start_pos_lst)

    for index, end_pos in enumerate(end_pos_lst):
        if end_pos != -1 and end_pos <= min_end_pos:
                min_end_pos = end_pos
                end_token1 = end_token_lst[index]
    if end_token1 != end_token:
        end_token1 = ""    

    return raw_text[max_start_pos:min_end_pos + len(end_token1)]


def re_sub_text(text):
    text = re.sub("'ll", " will", text)
    text = re.sub("n't", " not", text)

    # remove special str
    for pattern in rm_lst:
        text = re.sub(pattern, "", text)

    # replace ?, ! by .
    text = re.sub(r'[!?]', '.', text).strip()

    # # remove other punc and lower text
    # # text = re.sub(r'[^a-zA-Z0-9\s.,()!?&\']', ' ', text.lower()).strip()
    text = re.sub(r'[^a-zA-Z\s.]', ' ', text).strip()

    return text


def clean_sentence(sen):
    token = " [SEP]"
    sen = re.sub(r"\s+", " ", sen)
    sen = mask_entity(sen)
    sen = sen.replace(".", token)
    if not sen.endswith(token):
        sen = sen + token

    return sen.lower()

def split_and_concat_paragpraph(sliced_text):
    split_lst = sliced_text.split("\n")
    return_lst = []
    text = ""
    for sent in split_lst:
        if sent == "":
            if text != "":
                text = clean_sentence(text)
                return_lst.append(text)
                text = ""
        else:
            text = text + " " + sent

    if text.strip() != "":
        text = clean_sentence(text)
        return_lst.append(text)

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


# def preprocess_qa()

def clean_data(set_name, cleaned_data_path):
    logging.info(f"=== Cleaning {set_name} set and save to \'{cleaned_data_path}\'")

    dataset = load_dataset('narrativeqa', split=f'{set_name}[:1]')
    doc_id_set = set()
    for text in dataset:
        doc = text["document"]
        # doc
        if doc["id"] == "00fb61fa7bee266ad995e52190ebb73606b60b70" and doc["id"] not in doc_id_set:

            global entity_vocab
            global entity_vocab_per_doc

            entity_vocab_per_doc = {}
            doc_id_set.add(doc["id"])
            # logging.info(f"Preprocessing doc id {doc['id']}")
            clean_doc = preprocess_doc(doc["text"], doc["start"], doc["end"])
            for part in clean_doc:
                if part.strip() != "":
                    append_text(part.strip(), cleaned_data_path, f"{doc['id']}.txt")
            if len(entity_vocab_per_doc) > 0:
                entity_vocab[doc["id"]] = entity_vocab_per_doc


        qa
        if doc["id"] == "00fb61fa7bee266ad995e52190ebb73606b60b70" :
            qa_lst = []
            qa_lst.append(text["question"]["text"])
            for ans in text["answers"]:
                qa_lst.append(ans["text"])
            append_text("[SEP]".join(qa_lst), f"{cleaned_data_path}/qa", f"{doc['id']}.txt")
    

if __name__ == '__main__':

    ### Clean dataset
    dataset_list = ["train", "test", "validation"]
    for name in dataset_list:
        cleaned_data_path = f"./data/preprocessed/{name}"
        vocab_folder = "./data/vocab/"

        if not os.path.isdir(cleaned_data_path):
            clean_data(name, cleaned_data_path)
        if len(entity_vocab) > 0:
            write_pickle(entity_vocab, vocab_folder, "entities")





