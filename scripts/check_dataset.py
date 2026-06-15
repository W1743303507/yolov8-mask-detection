from collections import Counter
import os

counter = Counter()

label_dir = "dataset/labels/train"

for file in os.listdir(label_dir):
    if not file.endswith(".txt"):
        continue

    with open(os.path.join(label_dir, file), "r") as f:
        for line in f:
            cls = int(line.split()[0])
            counter[cls] += 1

print(counter)