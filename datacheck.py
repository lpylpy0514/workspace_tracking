import os
import cv2
import numpy


def traverse_folder(folder_path, a):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_path = os.path.join(root, file)
                # 在这里可以对图片进行操作
                # print(image_path)
                # 例如，打开图片
                image = cv2.imread(image_path)
                a = a + 1
                if a % 100 == 0:
                    print(a / 100)
                if numpy.any(numpy.isnan(image)):
                    print(image_path)
                    raise NotImplementedError
        for folder in dirs:
            # 继续遍历子文件夹
            a = traverse_folder(os.path.join(root, folder), a)
    return a
a = 0
folder_path = 'data'
traverse_folder(folder_path, a)