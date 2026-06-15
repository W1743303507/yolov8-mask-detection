from pathlib import Path

label_dir = Path("dataset_v2/labels")

mapping = {
    0: 1,  # without -> without
    1: 0,  # with -> with
    2: 2   # incorrect -> incorrect
}

for txt_file in label_dir.glob("*.txt"):

    lines = []

    with open(txt_file, "r") as f:
        for line in f:

            parts = line.strip().split()

            if len(parts) == 0:
                continue

            cls = int(parts[0])

            parts[0] = str(mapping[cls])

            lines.append(" ".join(parts))

    with open(txt_file, "w") as f:
        f.write("\n".join(lines))

print("标签转换完成")