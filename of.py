import cv2
import numpy as np
for i in range(1, 50):
    # 读取两张图片
    prev_frame = cv2.imread(f'/home/ymz/newdisk1/GOT10k/train/GOT-10k_Train_000006/{i:08}.jpg')
    next_frame = cv2.imread(f'/home/ymz/newdisk1/GOT10k/train/GOT-10k_Train_000006/{(i + 1):08}.jpg')

    # 将图像转换为灰度图
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 可视化光流
    h, w = prev_gray.shape
    flow_vis = np.zeros((h, w, 3), dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    flow_vis[..., 0] = 255
    flow_vis[..., 1] = np.uint8(ang*180/np.pi/2)
    flow_vis[..., 2] = np.uint8(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX))
    flow_vis = cv2.cvtColor(flow_vis, cv2.COLOR_HSV2BGR)

    # 显示结果
    cv2.imshow('Optical Flow', flow_vis)
    cv2.waitKey(0)
cv2.destroyAllWindows()