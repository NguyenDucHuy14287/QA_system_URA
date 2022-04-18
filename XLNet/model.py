# from transformers import XLNetTokenizer, XLNetModel
# import torch
#
#
# tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
# model = XLNetModel.from_pretrained("xlnet-base-cased")
#
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)
#
# last_hidden_states = outputs.last_hidden_state





# import torch
# from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
# # from keras.preprocessing.sequence import pad_sequences
# # from sklearn.model_selection import train_test_split
#
#
# from pytorch_transformers import XLNetModel, XLNetTokenizer, XLNetForSequenceClassification
# from pytorch_transformers import AdamW
#
# from tqdm import tqdm, trange
# import pandas as pd
# import io
# import numpy as np
# # import matplotlib.pyplot as plt
# # % matplotlib inline
#
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# n_gpu = torch.cuda.device_count()
# torch.cuda.get_device_name(0)






from transformers import XLNetTokenizer, XLNetForQuestionAnsweringSimple
import torch

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")

question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

answer_start_index = outputs.start_logits.argmax()
answer_end_index = outputs.end_logits.argmax()

predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
a = tokenizer.decode(predict_answer_tokens)
print(a)

