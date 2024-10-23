from transformers import GPT2Tokenizer, GPT2Model

cache_dir = "model_exp\gpt\model"
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path='gpt2', cache_dir=cache_dir, local_files_only=True)
model = GPT2Model.from_pretrained(pretrained_model_name_or_path='gpt2', cache_dir=cache_dir, local_files_only=True)
text = "Replace me by any text you'd like."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
print(tokenizer)
print(model)