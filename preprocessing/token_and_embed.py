import os
import sys

import spacy
from spacy.lang.en import English
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
from glove import Corpus, Glove
import re

from utils import *

def create_tokenizer():
    nlp = English()
    nlp.tokenizer = Tokenizer(nlp.vocab)

    prefixes = [r'''\'|\"''',]
    prefix_regex = spacy.util.compile_prefix_regex(prefixes)
    nlp.tokenizer.prefix_search = prefix_regex.search

    suffixes = [r'''\!|\,|\'s|\'d|\'m|n\'t|\'ll|\?|\"|\'''',]
    suffix_regex = spacy.util.compile_suffix_regex(suffixes)
    nlp.tokenizer.suffix_search = suffix_regex.search
    return nlp


def token_with_checking(text, tokenizer):
    pre_token = ""
    token_lst = []
    tokens = tokenizer(text)
    for token in tokens:
        if token.text != "\"" and token.text != "\'" and token.text.strip() != "":
            if pre_token != "[sep]" or token.text != "[sep]":
                token_lst.append(token.text)
                pre_token = token.text
    return token_lst


if __name__ == "__main__":
    ##### load dataset
    train_set = load_dataset_from_path("data/preprocessed/train")
    test_set = load_dataset_from_path("data/preprocessed/test")
    valid_set = load_dataset_from_path("data/preprocessed/validation")


    ##### create tokenizer
    spacy_tokenizer = create_tokenizer()


    ##### create data train
    context = []

    for doc in train_set:
        paragraph_lst = []
        for paragraph in doc["paragraph"]:
            token_lst = token_with_checking(paragraph, spacy_tokenizer)
            context.append(token_lst)
            paragraph_lst.append(token_lst)
        write_pickle(paragraph_lst, "./data/token/train", doc["id"])
        
        qa_lst = []
        for qa_text in doc["qa_text"]:
            token_lst = token_with_checking(qa_text, spacy_tokenizer)
            context.append(token_lst)
            qa_lst.append(token_lst)
        write_pickle(qa_lst, "./data/token/train/qa", doc["id"])

    for doc in test_set:
        paragraph_lst = []
        for paragraph in doc["paragraph"]:
            token_lst = token_with_checking(paragraph, spacy_tokenizer)
            context.append(token_lst)
            paragraph_lst.append(token_lst)
        write_pickle(paragraph_lst, "./data/token/test", doc["id"])
        
        qa_lst = []
        for qa_text in doc["qa_text"]:
            token_lst = token_with_checking(qa_text, spacy_tokenizer)
            context.append(token_lst)
            qa_lst.append(token_lst)
        write_pickle(qa_lst, "./data/token/test/qa", doc["id"])

    for doc in valid_set:
        paragraph_lst = []
        for paragraph in doc["paragraph"]:
            token_lst = token_with_checking(paragraph, spacy_tokenizer)
            context.append(token_lst)
            paragraph_lst.append(token_lst)
        write_pickle(paragraph_lst, "./data/token/validation", doc["id"])
        
        qa_lst = []
        for qa_text in doc["qa_text"]:
            token_lst = token_with_checking(qa_text, spacy_tokenizer)
            context.append(token_lst)
            qa_lst.append(token_lst)
        write_pickle(qa_lst, "./data/token/validation/qa", doc["id"])

    
    ##### Create glove and train glove
    glove_path = 'data/glove.model'

    if not os.path.isfile(glove_path):
        # Creating a corpus object
        corpus = Corpus() 

        # Training the corpus to generate the co-occurrence matrix which is used in GloVe
        corpus.fit(context, window=10)

        glove = Glove(no_components=400, learning_rate=0.01) 
        glove.fit(corpus.matrix, epochs=500, no_threads=4, verbose=True)
        glove.add_dictionary(corpus.dictionary)
        glove.save(glove_path)
    else:
        glove = Glove.load(glove_path)

    
    # #### Get embeded vector
    # print("Glove embeding dictionary:", glove.dictionary)

    # first_sentence = context[0]

    # for token in first_sentence:
    #     print("=== token:", token)
    #     print(glove.word_vectors[glove.dictionary[token]])  # embeded vector
    #     # print(glove.most_similar(token))                    # most similar token


    for doc in train_set:
        paragraph_embed_lst = []
        for paragraph in doc["paragraph"]:
            token_lst = token_with_checking(paragraph, spacy_tokenizer)
            paragraph_embed = [glove.word_vectors[glove.dictionary[token]] for token in token_lst]
            paragraph_embed_lst.append(paragraph_embed)
        write_pickle(paragraph_embed_lst, "./data/embed/train", doc["id"])
        
        qa_embed_lst = []
        for qa_text in doc["qa_text"]:
            token_lst = token_with_checking(qa_text, spacy_tokenizer)
            qa_embed = [glove.word_vectors[glove.dictionary[token]] for token in token_lst]
            qa_embed_lst.append(qa_embed)
        write_pickle(qa_embed_lst, "./data/embed/train/qa", doc["id"])

    for doc in test_set:
        paragraph_embed_lst = []
        for paragraph in doc["paragraph"]:
            token_lst = token_with_checking(paragraph, spacy_tokenizer)
            paragraph_embed = [glove.word_vectors[glove.dictionary[token]] for token in token_lst]
            paragraph_embed_lst.append(paragraph_embed)
        write_pickle(paragraph_embed_lst, "./data/embed/test", doc["id"])
        
        qa_embed_lst = []
        for qa_text in doc["qa_text"]:
            token_lst = token_with_checking(qa_text, spacy_tokenizer)
            qa_embed = [glove.word_vectors[glove.dictionary[token]] for token in token_lst]
            qa_embed_lst.append(qa_embed)
        write_pickle(qa_embed_lst, "./data/embed/test/qa", doc["id"])

    for doc in valid_set:
        paragraph_embed_lst = []
        for paragraph in doc["paragraph"]:
            token_lst = token_with_checking(paragraph, spacy_tokenizer)
            paragraph_embed = [glove.word_vectors[glove.dictionary[token]] for token in token_lst]
            paragraph_embed_lst.append(paragraph_embed)
        write_pickle(paragraph_embed_lst, "./data/embed/validation", doc["id"])
        
        qa_embed_lst = []
        for qa_text in doc["qa_text"]:
            token_lst = token_with_checking(qa_text, spacy_tokenizer)
            qa_embed = [glove.word_vectors[glove.dictionary[token]] for token in token_lst]
            qa_embed_lst.append(qa_embed)
        write_pickle(qa_embed_lst, "./data/embed/validation/qa", doc["id"])
