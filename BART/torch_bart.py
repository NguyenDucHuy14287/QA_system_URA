import torch
torch.cuda.empty_cache()

import pickle

def pickle_load(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

############### Load data
tokenized_datasets = pickle_load("tokenized_datasets")
tokenized_datasets.set_format("torch")

from torch.utils.data import DataLoader
train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(tokenized_datasets, batch_size=8)

############### Load model
model = pickle_load("model")

from torch.optim import AdamW
optimizer = AdamW(model.parameters(), lr=5e-5)

from transformers import get_scheduler
num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)


from datasets import load_metric
metric = load_metric("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()