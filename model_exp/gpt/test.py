from transformers import GPT2Tokenizer, GPT2Model

cache_dir = "E:\cudacode\CBWC-cuda\model_exp\gpt\model"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir)
model = GPT2Model.from_pretrained('gpt2', cache_dir=cache_dir)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
# print(output)