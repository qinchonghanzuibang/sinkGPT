import tiktoken
enc = tiktoken.get_encoding("gpt2")
# print enc size
print(f"GPT-2 词表大小：{enc.n_vocab}")
