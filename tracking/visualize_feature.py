import cv2
import torch
import math
import numpy as np

def visualize_feature(feature):
    B, N, C = feature.shape
    HW = int(math.sqrt(N))
    feature_show = feature.view(B, HW, HW, C)
    feature_show = feature_show * feature_show
    feature_show = feature_show.sum(dim=-1).cpu()
    print(feature_show.shape)
    # normalize
    feature_show = feature_show / feature_show.max()
    image = (feature_show * 255).permute(1, 2, 0).detach().numpy().astype(np.uint8)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_NEAREST)
    return image


def depreprocess(feature):
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
    image = feature.cpu() * std + mean
    image = (image * 255).squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)
    import cv2
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def cv_imshow(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    feature = torch.rand((1, 256, 768))
    image = visualize_feature(feature)
    cv_imshow(image)
    feature = torch.rand((1, 3, 256, 256))
    image = depreprocess(feature)
    print(image.shape)
    cv_imshow(image)