from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# tokenizer = AutoTokenizer.from_pretrained("Primer/bart-squad2")
# model = AutoModelForQuestionAnswering.from_pretrained("Primer/bart-squad2")
# model.to('cuda'); model.eval()

def answer(question, text, tokenizer, model):
    seq = '<s>' +  question + ' </s> </s> ' + text + ' </s>'
    tokens = tokenizer.encode_plus(seq, return_tensors='pt', padding='max_length', max_length=1024)
    input_ids = tokens['input_ids']#.to('cuda')
    attention_mask = tokens['attention_mask']#.to('cuda')
    start, end, _ = model(input_ids, attention_mask=attention_mask)
    start_idx = int(start.argmax().int())
    end_idx =  int(end.argmax().int())
    print(tokenizer.decode(input_ids[0, start_idx:end_idx]).strip())
   
