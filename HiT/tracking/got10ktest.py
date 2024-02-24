import os
import sys
import argparse
import cv2 as cv
import numpy as np
import pandas
import onnx
import onnxruntime
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
os.environ["CUDA_VISIBLE_DEVICES"]= "0"


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.test.tracker.vittrack_utils import sample_target
# for debug
import cv2
import os
import lib.models.HiT.levit_utils as utils
from lib.models.HiT import build_hit
from lib.test.tracker.vittrack_utils import Preprocessor
from lib.utils.box_ops import clip_box


class HiT:
    def __init__(self, net):
        self.net = net
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        # self.debug = False
        self.frame_id = 0
        # for save boxes from all queries
        self.z_dict1 = {}

    def initialize(self, image, info: dict):
        # forward the template once
        # info['init_bbox']: list [x0,y0,w,h] example: [367.0, 101.0, 41.0, 16.0]
        z_patch_arr, _ = sample_target(image, info['init_bbox'], 2.0, 128)
        self.template = self.preprocessor.process(z_patch_arr)
        self.state = info['init_bbox']
        self.frame_id = 0

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor = sample_target(image, self.state, 4.0, 256)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)
        ort_inputs = {'search': to_numpy(search).astype(np.float32),
                      'template': to_numpy(self.template).astype(np.float32)
                      }
        with torch.no_grad():
            ort_outs = self.net.run(None, ort_inputs)

        pred_boxes = torch.from_numpy(ort_outs[0]).view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(dim=0) * 256 / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        # self.state: list [x0,y0,w,h,] example: [365.4537048339844, 102.24719142913818, 47.13159942626953, 15.523386001586914]
        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * 256 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * 256 / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_tracker_class():
    return HiT


from math import sqrt
import tqdm
from lib.utils.box_ops import box_iou, box_xywh_to_xyxy

if __name__ == '__main__':
    data_dir = "/home/ymz/newdisk1/GOT10k/train/GOT-10k_Train_000005"
    gt_dir = data_dir + "/groundtruth.txt"
    gt = pandas.read_csv(gt_dir, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values

    imgs = []
    for file in os.listdir(data_dir):
        if file.endswith('jpg'):
            imgs.append(file)
    img_num = len(imgs)
    n = 1
    img = cv.imread(data_dir + f'/{n:08}.jpg')

    net_path = "/home/ymz/newdisk2/speed_test/HiT/checkpoints/HiT_Tiny/VT_ep1500.onnx"
    onnx_model = onnx.load(net_path)
    onnx.checker.check_model(onnx_model)
    with torch.no_grad():
        ort_session = onnxruntime.InferenceSession(net_path, providers=['CUDAExecutionProvider'])

    tracker = HiT(ort_session)

    x, y, w, h = gt[0]
    init_state = [x, y, w, h]  # left-top x y and w h
    frame = img

    def _build_init_info(box):
        return {'init_bbox': box}
    tracker.initialize(frame, _build_init_info(init_state))
    results = []
    for _ in tqdm.tqdm(range(img_num - 1)):
        n = n + 1
        frame = cv.imread(data_dir + f'/{n:08}.jpg')
        out = tracker.track(frame)
        results.append(out['target_bbox'])

    errors = []
    for i, result in enumerate(results):
        x_center = result[0] + 0.5 * result[2]
        y_center = result[1] + 0.5 * result[3]
        x_gt_center = gt[i + 1][0] + 0.5 * gt[i + 1][2]
        y_gt_center = gt[i + 1][1] + 0.5 * gt[i + 1][3]

        iou = box_iou(box_xywh_to_xyxy(torch.tensor(result)[None]), box_xywh_to_xyxy(torch.tensor(gt[i+1])[None]))[0]

        w_gt = gt[i + 1][2]
        h_gt = gt[i + 1][3]
        error = sqrt((x_center - x_gt_center) ** 2 + (y_center - y_gt_center) ** 2) / sqrt(w_gt * h_gt)
        if iou > 0.3:
            errors.append(error)
    print("error rate: ", sum(errors)/len(errors))
    print("tracker online percent: ", len(errors) / (img_num - 1))
    a = 1
