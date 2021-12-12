from langdetect import detect
import re
from bs4 import BeautifulSoup

URL_token = "<URL>"
NUMBER_token = "<NUMBER>"
COMMA_token = "<COMMA>" # ,
COLON_token = "<COLON>" # :
END_PUNC_TOKEN = "<END_PUNC>" # . ! ? -
START_BRACKET_token = "<START_BRACKET>" # {, (, [ -
END_BRACKET_token = "<END_BRACKET>" # }, ), ]

url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

start_token_extra_lst = ["produced by", "start of the project"]
end_token_extra_lst = ["start full license", "end of the project", "end of project"]

def write_text(text, path="data_text.txt"):
    f = open(path, "w", encoding="utf-8")
    f.write(text)
    f.close()

def append_text(text, path):
    file_object = open(path, 'a')
    file_object.write(text + "\n")
    file_object.close()

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
    end_pos_lst = [raw_text.find(i) for i in end_token_lst]

    max_start_pos = 0
    min_end_pos = len(raw_text)
    end_token1 = ""
    for index, start_pos in enumerate(start_pos_lst):
        if start_pos != -1 and start_pos >= max_start_pos:
                max_start_pos = start_pos

    for index, end_pos in enumerate(end_pos_lst):
        if end_pos != -1 and end_pos <=   min_end_pos:
                min_end_pos = end_pos
                end_token1 = end_token_lst[index]

    return raw_text[max_start_pos:min_end_pos + len(end_token1)]

def find_url(text):
    url = re.findall(url_regex, text)
    return [x[0] for x in url]

def preprocessing_text(text):
    # remove url
    text = re.sub(url_regex, "", text)

    # remove punc and lower text
    # text = re.sub('[!@#$%^&*()_+-=<>,/?~`"]+', '', text.lower()).strip()
    text = re.sub(r'[^a-zA-Z0-9\s.\']', ' ', text.lower()).strip()

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

