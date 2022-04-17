import torch
torch.cuda.empty_cache()

import pickle

def pickle_load(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

# ############### Load data
# tokenized_datasets = pickle_load("tokenized_datasets")
# tokenized_datasets = tokenized_datasets.remove_columns(["answers", "context", "question"])
# tokenized_datasets.set_format("torch")

# from torch.utils.data import DataLoader
# train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=8)
# eval_dataloader = DataLoader(tokenized_datasets, batch_size=8)

# ############### Load model
# model = pickle_load("model")
model = pickle_load("trained_model")
model.to(torch.device("cpu"))

# from torch.optim import AdamW
# optimizer = AdamW(model.parameters(), lr=5e-5)

# from transformers import get_scheduler
# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
# )

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# from tqdm.auto import tqdm
# progress_bar = tqdm(range(num_training_steps))

# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)


# def pickle_dump(data, file_path):
#     file = open(file_path, 'wb')
#     pickle.dump(data, file)
#     file.close()

# pickle_dump(model, 'trained_model')

tokenizer = pickle_load("tokenizer")
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
encoding = tokenizer(question, text, return_tensors='pt')
input_ids = encoding['input_ids']#.to(device)
attention_mask = encoding['attention_mask']#.to(device)

start_scores, end_scores = model(input_ids, attention_mask=attention_mask, output_attentions=False)[:2]

all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
answer = tokenizer.convert_tokens_to_ids(answer.split())
answer = tokenizer.decode(answer)
#answer => 'a nice puppet' 
print(answer)