import os
import xml.etree.ElementTree as ET

# =====================
# 路径
# =====================
XML_DIR = r"dataset800_raw/annotations"
IMG_DIR = r"dataset800_raw/images"

LABEL_DIR = r"dataset/labels"
IMAGE_OUT_DIR = r"dataset/images"

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

# =====================
# 类别映射
# =====================
classes = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}


def convert_box(size, box):
    w = size[0]
    h = size[1]

    xmin, ymin, xmax, ymax = box

    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0

    bw = xmax - xmin
    bh = ymax - ymin

    return (
        x_center / w,
        y_center / h,
        bw / w,
        bh / h
    )


for xml_file in os.listdir(XML_DIR):

    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(XML_DIR, xml_file)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    txt_name = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(LABEL_DIR, txt_name)

    with open(txt_path, "w") as f:

        for obj in root.findall("object"):

            cls_name = obj.find("name").text

            if cls_name not in classes:
                continue

            cls_id = classes[cls_name]

            box = obj.find("bndbox")

            xmin = float(box.find("xmin").text)
            ymin = float(box.find("ymin").text)
            xmax = float(box.find("xmax").text)
            ymax = float(box.find("ymax").text)

            x, y, w, h = convert_box(
                (width, height),
                (xmin, ymin, xmax, ymax)
            )

            f.write(
                f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
            )

print("XML 转 YOLO 完成！")