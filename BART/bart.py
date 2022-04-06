import pickle
def pickle_load(file_path):
    file = open(file_path, 'rb')
    data = pickle.load(file)
    file.close()
    return data

def pickle_dump(data, file_path):
    file = open(file_path, 'wb')
    pickle.dump(data, file)
    file.close()

# ### prepare dataset
# from datasets import Dataset
# train_set = pickle_load("sample_train")
# train_set_dict = {
#     'answers': [0] * len(train_set), 
#     'context': [0] * len(train_set), 
#     'question': [0] * len(train_set)
# }

# for i, data in enumerate(train_set):
#     train_set_dict['answers'][i] = data['answers']
#     train_set_dict['context'][i] = data['context']
#     train_set_dict['question'][i] = data['question']

# dataset = Dataset.from_dict(train_set_dict)


### preprocess
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('a-ware/bart-squadv2')
tokenizer = pickle_load("tokenizer")

# max_length = 512 # The maximum length of a feature (question and context)
# doc_stride = 128 # The authorized overlap between two part of the context when splitting it is needed.
# pad_on_right = tokenizer.padding_side == "right"

# def prepare_train_features(examples):
#     # Some of the questions have lots of whitespace on the left, which is not useful and will make the
#     # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
#     # left whitespace
#     examples["question"] = [q.lstrip() for q in examples["question"]]

#     # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
#     # in one example possible giving several features when a context is long, each of those features having a
#     # context that overlaps a bit the context of the previous feature.
#     tokenized_examples = tokenizer(
#         examples["question" if pad_on_right else "context"],
#         examples["context" if pad_on_right else "question"],
#         truncation="only_second" if pad_on_right else "only_first",
#         max_length=max_length,
#         stride=doc_stride,
#         return_overflowing_tokens=True,
#         return_offsets_mapping=True,
#         padding="max_length",
#     )

#     # Since one example might give us several features if it has a long context, we need a map from a feature to
#     # its corresponding example. This key gives us just that.
#     sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
#     # The offset mappings will give us a map from token to character position in the original context. This will
#     # help us compute the start_positions and end_positions.
#     offset_mapping = tokenized_examples.pop("offset_mapping")

#     # Let's label those examples!
#     tokenized_examples["start_positions"] = []
#     tokenized_examples["end_positions"] = []

#     for i, offsets in enumerate(offset_mapping):
#         # We will label impossible answers with the index of the CLS token.
#         input_ids = tokenized_examples["input_ids"][i]
#         cls_index = input_ids.index(tokenizer.cls_token_id)

#         # Grab the sequence corresponding to that example (to know what is the context and what is the question).
#         sequence_ids = tokenized_examples.sequence_ids(i)

#         # One example can give several spans, this is the index of the example containing this span of text.
#         sample_index = sample_mapping[i]
#         answers = examples["answers"][sample_index]
#         # If no answers are given, set the cls_index as answer.
#         if len(answers["answer_start"]) == 0:
#             tokenized_examples["start_positions"].append(cls_index)
#             tokenized_examples["end_positions"].append(cls_index)
#         else:
#             # Start/end character index of the answer in the text.
#             start_char = answers["answer_start"][0]
#             end_char = start_char + len(answers["text"][0])

#             # Start token index of the current span in the text.
#             token_start_index = 0
#             while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
#                 token_start_index += 1

#             # End token index of the current span in the text.
#             token_end_index = len(input_ids) - 1
#             while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
#                 token_end_index -= 1

#             # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
#             if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
#                 tokenized_examples["start_positions"].append(cls_index)
#                 tokenized_examples["end_positions"].append(cls_index)
#             else:
#                 # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
#                 # Note: we could go after the last offset if the answer is the last word (edge case).
#                 while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
#                     token_start_index += 1
#                 tokenized_examples["start_positions"].append(token_start_index - 1)
#                 while offsets[token_end_index][1] >= end_char:
#                     token_end_index -= 1
#                 tokenized_examples["end_positions"].append(token_end_index + 1)

#     return tokenized_examples

# tokenized_datasets = dataset.map(prepare_train_features, batched=True)
tokenized_datasets = pickle_load("tokenized_datasets")


### modeling
# from transformers import AutoModelForQuestionAnswering, BartForQuestionAnswering
# model = BartForQuestionAnswering.from_pretrained('a-ware/bart-squadv2')
model = pickle_load("model")

from transformers import TrainingArguments, Trainer
args = TrainingArguments(
    f"bart-squadv2-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

from transformers import default_data_collator
data_collator = default_data_collator

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)


print(trainer)