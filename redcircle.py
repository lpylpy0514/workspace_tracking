import cv2 as cv
import torch
import pandas
import os
import numpy as np
from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target

import copy


def draw(img, mode, transparency, pt1, pt2, color, thickness):
    assert transparency <= 1 and transparency >= 0, 'transparency is not proper.'
    img0 = copy.deepcopy(img)
    if mode == 'rect':
        cv.rectangle(img, pt1, pt2, color, thickness)
    elif mode == 'ellipse':
        xcenter = int(pt1.x + pt2.x) // 2
        ycenter = int(pt1.y + pt2.y) // 2
        w = int(pt2.x - pt1.x)
        h = int(pt2.y - pt1.y)
        assert w > 0 and h > 0, 'point is not proper'
        cv.ellipse(img, (xcenter, ycenter), (w // 2, h // 2), 0, 0, 360, color, thickness)
    elif mode == 'line':
        cv.line(img, pt1, pt2, color, thickness)
    else:
        raise NotImplementedError
    img = cv.addWeighted(img, transparency, img0, (1 - transparency), 0)
    return img


if __name__ == '__main__':

    for i in range(1, 180):
        data_dir = f"/home/ymz/newdisk1/GOT10k/test/GOT-10k_Test_{i:06}"
        gt_dir = data_dir + "/groundtruth.txt"
        gt = pandas.read_csv(gt_dir, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values

        imgs = []
        for file in os.listdir(data_dir):
            if file.endswith('jpg'):
                imgs.append(file)
        img_num = len(imgs)
        n = 1
        img = cv.imread(data_dir + f'/{n:08}.jpg')
        img0 = cv.imread(data_dir + f'/{n:08}.jpg')
        k = 1
        x1 = gt[0][0]
        y1 = gt[0][1]
        cx = int(gt[0][0] + 0.5 * gt[0][2])
        cy = int(gt[0][1] + 0.5 * gt[0][3])
        w = int(gt[0][2] * k)
        h = int(gt[0][3] * k)
        import copy
        imgtemp = copy.deepcopy(img)
        cv.rectangle(img, (int(gt[0][0]), int(gt[0][1])), (int(gt[0][0] + gt[0][2]), int(gt[0][1] + gt[0][3])),
                     color=(0, 0, 255), thickness=3)
        k = 0.4
        img = cv.addWeighted(img, k, imgtemp, (1 - k), 0)
        # cv.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, color=(0, 0, 255), thickness=3)
        print(img.shape)

        pre = Preprocessor()
        x_patch_arr, resize_factor, x_amask_arr = sample_target(img, [x1, y1, w, h], 4,
                                                                output_sz=640)  # (x1, y1, w, h)
        search = pre.process(x_patch_arr, x_amask_arr)

        z_patch_arr, resize_factor, z_amask_arr = sample_target(img, [x1, y1, w, h], 2,
                                                                output_sz=128)  # (x1, y1, w, h)
        template = pre.process(z_patch_arr, z_amask_arr)

        z_patch_arr0, resize_factor, z_amask_arr = sample_target(img0, [x1, y1, w, h], 2,
                                                                output_sz=128)  # (x1, y1, w, h)
        template = pre.process(z_patch_arr, z_amask_arr)

        cv.imshow("search", x_patch_arr)
        cv.imshow("template", z_patch_arr)
        # img_resize = cv.resize(z_patch_arr, (512, 512))
        cv.imshow("template0", z_patch_arr0)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # img = cv.resize(img, (960, 540))
        # cv.imshow("img", img)
        cv.waitKey(0)
    cv.destroyAllWindows()
