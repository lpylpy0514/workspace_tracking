import copy
import cv2 as cv
import torch


def draw(img, mode, transparency, pt1, pt2, color, thickness):
    assert 1 >= transparency >= 0, 'transparency is not proper.'
    img0 = copy.deepcopy(img)
    if mode == 'rect':
        cv.rectangle(img, pt1, pt2, color, thickness)
    elif mode == 'ellipse':
        x_center = int(pt1[0] + pt2[0]) // 2
        y_center = int(pt1[1] + pt2[1]) // 2
        w = int(pt2[0] - pt1[0])
        h = int(pt2[1] - pt1[1])
        assert w > 0 and h > 0, 'point is not proper, pt2 is the right-bottom point'
        cv.ellipse(img, (x_center, y_center), (w // 2, h // 2), 0, 0, 360, color, thickness)
    elif mode == 'line':
        cv.line(img, pt1, pt2, color, thickness)
    else:
        raise NotImplementedError
    img = cv.addWeighted(img, transparency, img0, (1 - transparency), 0)
    return img


class Draw(torch.nn.Module):
    def __init__(self, template_size=128, template_factor=2, thickness=3, mode='rect', colormode='fixedcolor'):
        # template_size = template_factor * sqrt(h * w)
        super().__init__()
        self.mode = mode
        self.colormode = colormode
        if colormode == 'fixedcolor':
            self.color = torch.nn.Parameter(torch.zeros(3))
            self.transparency = torch.nn.Parameter(torch.zeros(1))
            self.transparency.data.fill_(0.5)
        elif colormode == 'learncolor':
            from lib.models.ostrack.preprocess import build_preprocess
            self.color_generate = build_preprocess(template_size=128, patch_size=16, embed_dim=768, depth=1, output_dim=4)

        self.template_size = template_size
        self.template_factor = template_factor
        self.center = (template_size - 1) / 2
        self.thickness = thickness

        # generate mesh-grid
        self.indice = torch.arange(0, template_size).view(-1, 1)
        coord_x = self.indice.repeat((self.template_size, 1)).view(self.template_size, self.template_size).float()
        coord_y = self.indice.repeat((1, self.template_size)).view(self.template_size, self.template_size).float()
        self.register_buffer("coord_x", coord_x)
        self.register_buffer("coord_y", coord_y)

    def forward(self, template, gt):
        # mask
        # learn color based on template

        B = template.shape[0]
        x, y, w, h = gt.unbind(1)
        template_w = self.template_size / self.template_factor * torch.sqrt(w / h)
        template_h = self.template_size / self.template_factor * torch.sqrt(h / w)
        template_draw = copy.deepcopy(template)
        x1 = (self.center - template_w / 2).view(B, 1, 1)
        y1 = (self.center - template_h / 2).view(B, 1, 1)
        x2 = (self.center + template_w / 2).view(B, 1, 1)
        y2 = (self.center + template_h / 2).view(B, 1, 1)

        if self.mode == 'rect':
            index = (x2 + self.thickness / 2 > self.coord_x) & (self.coord_x > x1 - self.thickness / 2) \
                & (y2 + self.thickness / 2 > self.coord_y) & (self.coord_y > y1 - self.thickness / 2) \
                & ~((x2 - self.thickness / 2 > self.coord_x) & (self.coord_x > x1 + self.thickness / 2)
                & (y2 - self.thickness / 2 > self.coord_y) & (self.coord_y > y1 + self.thickness / 2))
        elif self.mode == 'mask':
            index = (x2 > self.coord_x) & (self.coord_x > x1) & (y2 > self.coord_y) & (self.coord_y > y1)
        else:
            raise NotImplementedError

        if self.colormode == 'fixedcolor':
            template_draw = template_draw.permute(0, 2, 3, 1)
            template_draw[index] = self.color
            template_draw = template_draw.permute(0, 3, 1, 2)
            output = template_draw * (1 - self.transparency) + template * self.transparency
        elif self.colormode == 'learncolor':
            color, transparency = self.color_generate(template) # color[B, 3], transparency[B], index[B, template_size, template_size]
            template_draw = template_draw.permute(0, 2, 3, 1)
            color = color.view(B, 1, 1, 3).repeat(1, self.template_size, self.template_size, 1)
            template_draw[index] = color[index]
            template_draw = template_draw.permute(0, 3, 1, 2)
            transparency = transparency.view((B, 1, 1, 1))
            output = template_draw * (1 - transparency) + template * transparency
        else:
            raise NotImplementedError
        # image = depreprocess(output)
        # import cv2
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # image = cv2.resize(image, (512, 512))
        # cv2.imshow('image', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return output


def depreprocess(feature):
    mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1))
    std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1))
    image = feature.cpu() * std + mean
    image = (image * 255).squeeze(0).permute(1, 2, 0).detach().numpy().astype(np.uint8)
    return image


from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.processing_utils import sample_target
import numpy as np
import pandas


if __name__ == '__main__':
    for i in range(1, 180):
        n = 1
        # data_dir = f"/home/ymz/newdisk1/GOT10k/test/GOT-10k_Test_{i:06}"
        # img_dir = data_dir + f'/{n:08}.jpg'
        # image = cv.imread(img_dir)
        # image = draw(image, mode='ellipse', transparency=0.2, pt1=(10, 10), pt2=(10, 100), color=(0, 0, 255), thickness=3)
        # cv.imshow('img', image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # fun = Draw(template_size=7, thickness=2)
        # template = torch.randn((2, 3, 7, 7))
        # gt = torch.tensor(((0, 0, 1.1, 1), (0, 0, 1, 1.1)))
        # fun(template, gt)
        data_dir = f"/home/ymz/newdisk1/GOT10k/train/GOT-10k_Train_{i:06}"
        # data_dir = f"/home/ymz/newdisk1/GOT10k/test/GOT-10k_Test_{i:06}"
        gt_dir = data_dir + "/groundtruth.txt"
        gt = pandas.read_csv(gt_dir, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        img = cv.imread(data_dir + f'/{n:08}.jpg')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        x1 = gt[0][0]
        y1 = gt[0][1]
        w = gt[0][2]
        h = gt[0][3]
        pre = Preprocessor()
        z_patch_arr, resize_factor, z_amask_arr = sample_target(img, [x1, y1, w, h], 2,
                                                                output_sz=128)  # (x1, y1, w, h)
        template = pre.process(z_patch_arr, z_amask_arr).tensors.cpu()
        template = torch.rand((2, 3, 128, 128))
        fun = Draw(template_size=128, thickness=3, mode='mask', colormode='learncolor')
        # fun.transparency.data = torch.tensor([0.8169])
        # fun.color.data = torch.tensor([-0.4978,  2.2672, -4.3935])
        template = template.cuda()
        gt = torch.tensor(gt[0:2]).cuda()
        fun = fun.cuda()
        template = fun(template, gt)
        # recover
        image = depreprocess(template)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image = cv.resize(image, (512, 512))
        cv.imshow('img', image)
        cv.waitKey(0)
    cv.destroyAllWindows()
