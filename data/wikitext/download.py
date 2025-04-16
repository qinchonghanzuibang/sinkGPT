from datasets import load_dataset

# Load the WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Write the WikiText-2 data to a file
with open("data/wikitext/wikitext2.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        if example["text"].strip():
            f.write(example["text"].strip() + "\n")
print("data/wikitext/wikitext2.txt file generated.")

# Load the WikiText-103 dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Write the WikiText-103 data to a file
with open("data/wikitext/wikitext103.txt", "w", encoding="utf-8") as f:
    for example in dataset:
        if example["text"].strip():
            f.write(example["text"].strip() + "\n")
print("data/wikitext/wikitext103.txt file generated.")