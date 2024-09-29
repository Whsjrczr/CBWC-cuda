from transformers import GPT2Tokenizer, GPT2Model

cache_dir = "E:\cudacode\CBWC-cuda\model_exp\gpt\model"
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', cache_dir=cache_dir, local_files_only=True)
model = GPT2Model.from_pretrained('gpt2', cache_dir=cache_dir, local_files_only=True)

print(model)
print(tokenizer)