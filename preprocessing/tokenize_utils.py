import os

from tokenizers import BertWordPieceTokenizer, Tokenizer, normalizers, pre_tokenizers
from tokenizers.normalizers import NFD, StripAccents
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Digits, Whitespace, Punctuation

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
        file_path_lst = [f"{data_path}/{file_name}" for file_name in os.listdir(data_path)]
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



