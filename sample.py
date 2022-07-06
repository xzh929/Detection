from PIL import Image
from pathlib import Path
import math
import xml.etree.ElementTree as et

img_path = Path(r"D:\data\oridata\images")
label_path = Path(r"D:\data\oridata\label")
save_txt_path = Path(r"D:\code\py\datasets\mydata\labels\train")
save_img_path = Path(r"D:\code\py\datasets\mydata\images\train")
cls_dict = {"键盘": 0, "鼠标": 1}


def img_process(img_resize):
    for file_name, label_xml in zip(img_path.iterdir(), label_path.iterdir()):
        img = Image.open(file_name).convert(mode="RGB")
        w, h = img.size
        scale = img_resize / max(w, h)
        new_w, new_h = math.floor(w * scale), math.floor(h * scale)
        img = img.resize((new_w, new_h))
        if len(file_name.stem) == 3:
            save_file_name = '0' + file_name.stem
        else:
            save_file_name = file_name.stem
        img.save(save_img_path.joinpath(save_file_name + '.jpg'))

        tree = et.parse(label_xml)
        root = tree.getroot()
        with open(save_txt_path.joinpath(save_file_name + '.txt'), "w", encoding="utf-8") as f:
            for elem in root.iter(tag="object"):
                name = elem.findtext("name")
                cls = cls_dict[name]
                x1, y1 = coordinate_process(int(elem.findtext("bndbox/xmin")), int(elem.findtext("bndbox/ymin")), w, h)
                x1, y1 = x1 * scale, y1 * scale
                x2, y2 = coordinate_process(int(elem.findtext("bndbox/xmax")), int(elem.findtext("bndbox/ymax")), w, h)
                x2, y2 = x2 * scale, y2 * scale
                cx = (x2 + x1) / 2 / img_resize
                cy = (y2 + y1) / 2 / img_resize
                width = (x2 - x1) / img_resize
                height = (y2 - y1) / img_resize
                f.write("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(cls, cx, cy, width, height))
                print(file_name, cls, cx, cy, width, height)


def coordinate_process(x, y, w, h):
    xc, yc = x, y
    if x < 0:
        xc = 0
    if y < 0:
        yc = 0
    if x > w:
        xc = w
    if y > h:
        yc = h
    return xc, yc


if __name__ == '__main__':
    img_process(416)
