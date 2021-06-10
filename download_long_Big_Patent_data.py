from datasets import load_dataset
import random
import json

dataset = load_dataset("big_patent")

text_sizes = []
summary_sizes = []
subset = "train"

files = [
        "train.json",
        "val.json",
        "test.json",
    ]

fps = {file: open(file, "wt") for file in files}

num_rows = dataset[subset].num_rows
num_train = 0
num_test = 0
num_val = 0

for i in range(num_rows):
    text = dataset[subset][i]['description']
    text_length = len(text.split(" "))
    if text_length > 4000:
        r = random.random()
        if 0.9 < r and num_test < 2000:
            group = "test"
            num_test += 1
        elif 0.8 < r < 0.9 and num_val < 2000:
            group = "val"
            num_val += 1
        else:
            group = "train"
            num_train += 1

        summary = dataset[subset][i]['abstract']
        summary_size = len(summary.split(" "))

        json_obj = {"text": text, "summary": summary}

        json.dump(
            json_obj,
            fps[group + ".json"],
            sort_keys=False,
            indent=None,
            ensure_ascii=False,
        )
        fps[group + ".json"].write("\n")

        text_sizes.append(text_length)
        summary_sizes.append(summary_size)

for fp in fps.values():
    fp.close()
