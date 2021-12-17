import os
from langdetect import detect
import re
from bs4 import BeautifulSoup

url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
rm_lst = [url_regex, "\t"]

start_token_extra_lst = ["produced by", "start of the project", "start of this project"]
end_token_extra_lst = ["start full license", "end of the project", "end of project"]

def write_text(text, path="data_text.txt"):
    f = open(path, "w", encoding="utf-8")
    f.write(text)
    f.close()

def append_text(text, path):
    f = open(path, 'a', encoding="utf-8")
    f.write(text + "\n")
    f.close()

def read_text(path):
    f = open(path, "r", encoding="utf-8")
    text = f.read()
    f.close()
    return text

def is_english(full_text, max_len=150):
    if len(full_text) < max_len:
        max_len = len(full_text)

    if detect(full_text[:max_len]) == "en":
        return True
    else:
        return False
    # return detect(full_text[:max_len])

def is_html(text, thresh_hold=2):
    html_tag = BeautifulSoup(text, "html.parser").find()
    if html_tag is None:
        return False
    elif len(html_tag) <= thresh_hold:
        return False
    else:
        return True

def slice_content(raw_text, start_token, end_token):
    start_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in start_token_extra_lst + [start_token]]
    end_token_lst = [re.sub('[\W ]+', ' ', i.lower()).strip() for i in end_token_extra_lst + [end_token]]

    start_pos_lst = [raw_text.find(i) for i in start_token_lst]
    end_pos_lst = [raw_text.rfind(i) for i in end_token_lst]

    max_start_pos = 0
    min_end_pos = len(raw_text)
    end_token1 = ""
    for index, start_pos in enumerate(start_pos_lst):
        if start_pos >= max_start_pos:
                max_start_pos = start_pos

    for index, end_pos in enumerate(end_pos_lst):
        if end_pos != -1 and end_pos <= min_end_pos:
                min_end_pos = end_pos
                end_token1 = end_token_lst[index]

    return raw_text[max_start_pos:min_end_pos + len(end_token1)]

def find_url(text):
    url = re.findall(url_regex, text)
    return [x[0] for x in url]

def re_sub_text(text):
    # remove special str
    for pattern in rm_lst:
        text = re.sub(pattern, "", text)

    # replace ?, ! by .
    text = re.sub(r'[!?]', '.', text).strip()

    # remove other punc and lower text
    text = re.sub(r'[^a-zA-Z0-9\s.,\']', ' ', text.lower()).strip()

    return text

def split_and_concat_paragpraph(slice_text):
    split_lst = slice_text.split("\n")
    return_lst = []
    text = ""
    for sent in split_lst:
        if sent == "":
            if text != "":
                return_lst.append(text)
                text = ""
        else:
            text = text + " " + sent

    if text != "":
        return_lst.append(text)

    return return_lst

def clean_paragpraph(paragraphs):
    sentences_lst = []
    for para in paragraphs:
        tokens_lst = []
        sentences = para.split(".")
        for i in sentences:
            tokens = list(filter(lambda x: x != "", i.split(" ")))
            if len(tokens) > 0:
                if len(tokens) == 1 and tokens[0].isnumeric():
                    pass
                else:
                    tokens_lst.append(tokens)
        if len(tokens_lst) != 0:
                sentences_lst.append(tokens_lst)

    def join_tokens(sent_lst):
        full_sent_lst = [' ' + ' '.join(j) for j in sent_lst]
        return '.'.join(full_sent_lst)[1:] + "."

    return_para = [join_tokens(i) for i in sentences_lst]
    return return_para

def preprocess(raw_text, start_token, stop_token):
    if is_html(raw_text):
        raw_text = BeautifulSoup(raw_text, "lxml").text

    preprocess_text = re_sub_text(raw_text)
    slice_text = slice_content(preprocess_text, start_token, stop_token)
    paragraph = split_and_concat_paragpraph(slice_text)
    return clean_paragpraph(paragraph)

def load_dataset_from_path(data_path):
    file_lst = os.listdir(data_path)
    doc = []
    for file_name in file_lst:
        full_text = read_text(f"{data_path}/{file_name}")
        paragraph = full_text.split("\n")
        doc.append(paragraph[:-1])
    return doc


