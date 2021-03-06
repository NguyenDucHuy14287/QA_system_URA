import os
import pandas as pd 
from tokenizers import BertWordPieceTokenizer, Tokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Digits, Whitespace, Punctuation

from utils import load_dataset_from_path


class hf_tokenizer(): # hugging-face tokenizer
    def __init__(self, from_bert_pretrained=False, pretrained_bert_file="data/bert-base-uncased-vocab.txt"):
        self.tokenizer = None
        if from_bert_pretrained:
            if os.path.isfile(pretrained_bert_file):
                print(f"=== [INFO] Load tokenizer vocab from \'{pretrained_bert_file}\'.")
                self.tokenizer = BertWordPieceTokenizer(pretrained_bert_file, lowercase=True)
            else:
                print(f"=== [ERROR] Can not found \'{pretrained_bert_file}\'.")
        else:
            print(f"=== [INFO] Create normal tokenizer with default config.")
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents()])
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation(), Digits()])
            self.trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

    def train_tokenizer(self, data_path):
        print(f"Training tokenizer from \'{data_path}\'.")
        file_path_lst = [f"{data_path}/{file_name}" for file_name in os.listdir(data_path) if file_name[-4:] == ".txt"]
        self.tokenizer.train(file_path_lst, self.trainer)

    def save_tokenizer(self, tokenizer_path_file="data/tokenizer_vocab.txt"):
        self.tokenizer.save(tokenizer_path_file)
        print(f"=== [INFO] Save tokenizer vocab to \'{tokenizer_path_file}\'.")

    def load_tokenizer(self, tokenizer_path_file="data/tokenizer_vocab.txt"):
        self.tokenizer = Tokenizer.from_file(tokenizer_path_file)
        print(f"=== [INFO] Load tokenizer vocab from \'{tokenizer_path_file}\'.")

    def encode_sentence(self, sentence):
        return self.tokenizer.encode(sentence)

    def encode_sentence_lst(self, sentence_lst):
        return self.tokenizer.encode_batch(sentence_lst)
        

if __name__ == "__main__":

    use_bert_pretrained_tokenize = True

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
    train_set = load_dataset_from_path("./data/preprocessed/train")           # 1099
    test_set = load_dataset_from_path("./data/preprocessed/test")             # 354
    valid_set = load_dataset_from_path("./data/preprocessed/validation")      # 115


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
    # print(qa_token)