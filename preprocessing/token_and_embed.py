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
    nlp.tokenizer.token_match =re.compile(r'\S+').match
    return nlp


if __name__ == "__main__":
    ##### load dataset
    train_set = load_dataset_from_path("data/preprocessed/train")
    test_set = load_dataset_from_path("data/preprocessed/test")
    valid_set = load_dataset_from_path("data/preprocessed/validation")


    ##### create tokenizer
    spacy_tokenizer = create_tokenizer()


    ##### create data train
    context = []
    
    dataset_lst = [train_set, test_set, valid_set]
    for dataset in dataset_lst:
        for doc in dataset:
            for paragraph in doc["paragraph"]:
                tokens = spacy_tokenizer(paragraph)
                token_lst = [token.text for token in tokens if token.text.strip() != ""]
                context.append(token_lst)
            
            for qa_text in doc["qa_text"]:
                tokens = spacy_tokenizer(qa_text)
                token_lst = [token.text for token in tokens if token.text.strip() != ""]
                context.append(token_lst)

    
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

    
    #### Get embeded vector
    print("Glove embeding dictionary:", glove.dictionary)

    first_sentence = context[0]

    for token in first_sentence:
        print("=== token:", token)
        print(glove.word_vectors[glove.dictionary[token]])  # embeded vector
        # print(glove.most_similar(token))                    # most similar token


