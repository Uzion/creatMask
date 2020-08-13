import os
import json
import numpy as np
import argparse
import cv2

# python create_mask.py -o mask1

IMAGE_FOLDER = "./data/"
MASK_FOLDER = "./mask"
PATH_ANNOTATION_JSON = "data-annotation.json"

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str)
parser.add_argument("-o", "--output", type=str)
parser.add_argument("-j", "--jason", type=str)
args = parser.parse_args()

if args.input:
    IMAGE_FOLDER = args.input + '/'
    print("1")
if args.output:
    MASK_FOLDER = args.output + '/'
    print("2")
if args.jason:
    PATH_ANNOTATION_JSON = args.jason
    print("3")
# 加载VIA导出的json文件
annotations = json.load(open(PATH_ANNOTATION_JSON, "r"))
# imgs = annotations["_via_img_metadata"]
imgs = annotations
lenghth = len(annotations)
# print(imgs)
for imgId in imgs:
    filename = imgs[imgId]["filename"]
    regions = imgs[imgId]["regions"]
    if len(regions) <= 0:
        continue

    mask_num = len(regions)

    # 图片路径
    image_path = os.path.join(IMAGE_FOLDER, filename)
    print(filename)
    # 读出图片，目的是获取到宽高信息
    image = cv2.imread(image_path)  # image = skimage.io.imread
    height, width = image.shape[:2]

    # 创建空的mask
    maskImage = np.zeros((height, width), dtype=np.uint8)
    for index in range(mask_num):
        # 取出第一个标注的类别，本例只标注了一个物件
        polygons = regions[index]["shape_attributes"]
        countOfPoints = len(polygons["all_points_x"])
        points = [None] * countOfPoints
        for i in range(countOfPoints):
            x = int(polygons["all_points_x"][i])
            y = int(polygons["all_points_y"][i])
            points[i] = (x, y)

        contours = np.array(points)

        # 遍历图片所有坐标
        for i in range(width):
            for j in range(height):
                if cv2.pointPolygonTest(contours, (i, j), False) > 0:
                    maskImage[j, i] = 1

    savePath = MASK_FOLDER + filename
    # 保存mask
    cv2.imwrite(savePath, maskImage)

print("all done!")
