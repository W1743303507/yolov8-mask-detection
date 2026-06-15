import os
import shutil
import random

random.seed(42)

IMG_DIR = "dataset800_raw/images"
LBL_DIR = "dataset/labels"

TRAIN_IMG = "dataset/images/train"
VAL_IMG = "dataset/images/val"

TRAIN_LBL = "dataset/labels/train"
VAL_LBL = "dataset/labels/val"

for p in [
    TRAIN_IMG,
    VAL_IMG,
    TRAIN_LBL,
    VAL_LBL
]:
    os.makedirs(p, exist_ok=True)

images = []

for f in os.listdir(IMG_DIR):
    if f.endswith(".png"):
        images.append(f)

random.shuffle(images)

split = int(len(images) * 0.8)

train_imgs = images[:split]
val_imgs = images[split:]

print("Train:", len(train_imgs))
print("Val:", len(val_imgs))

for img in train_imgs:

    shutil.copy(
        os.path.join(IMG_DIR, img),
        os.path.join(TRAIN_IMG, img)
    )

    txt = img.replace(".png", ".txt")

    shutil.copy(
        os.path.join(LBL_DIR, txt),
        os.path.join(TRAIN_LBL, txt)
    )

for img in val_imgs:

    shutil.copy(
        os.path.join(IMG_DIR, img),
        os.path.join(VAL_IMG, img)
    )

    txt = img.replace(".png", ".txt")

    shutil.copy(
        os.path.join(LBL_DIR, txt),
        os.path.join(VAL_LBL, txt)
    )

print("划分完成")