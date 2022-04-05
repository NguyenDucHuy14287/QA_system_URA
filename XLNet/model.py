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






from transformers import XLNetTokenizer, XLNetForQuestionAnswering
import torch

tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetForQuestionAnswering.from_pretrained("xlnet-base-cased")

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(
    0
)  # Batch size 1
start_positions = torch.tensor([1])
end_positions = torch.tensor([3])
outputs = model(input_ids, start_positions=start_positions, end_positions=end_positions)

loss = outputs.loss
print(outputs)

