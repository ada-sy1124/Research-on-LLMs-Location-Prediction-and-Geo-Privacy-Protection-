
from datasets import load_from_disk

YES_DIR = "/Applications/Documents/geoai/random_streetview_gemini/YES"
SAMPLE_INDEX = 0

dataset = load_from_disk(YES_DIR)
sample = dataset[SAMPLE_INDEX]

print("样本索引:", SAMPLE_INDEX)
print("所有列名:", dataset.column_names)

for column in dataset.column_names:
    if column == "image":
        print(f"{column}: <image>")
    else:
        print(f"{column}: {sample[column]}")

if "image" in sample and sample["image"] is not None:
    sample["image"].show()
