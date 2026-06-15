from pathlib import Path
import shutil

src_img = Path("dataset_v2/images")
src_lab = Path("dataset_v2/labels")

dst_img = Path("dataset_v3/images/train")
dst_lab = Path("dataset_v3/labels/train")

count = 0

for img in src_img.glob("*.*"):

    if img.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    txt = src_lab / f"{img.stem}.txt"

    if not txt.exists():
        continue

    shutil.copy2(img, dst_img / img.name)
    shutil.copy2(txt, dst_lab / txt.name)

    count += 1

print(f"成功导入 {count} 个样本")