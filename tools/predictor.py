from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import numpy as np
import random
import argparse
import pandas as pd
import cv2
from config import cfg
from network import Network
from torchvision import transforms
from PIL import Image


# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def make_transformer():
    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transformer


class Classifier:
    def __init__(self, config, model_path='./models/resnet18.pth', device='cpu'):
        super(Classifier, self).__init__()
        self.cfg = config
        self.model = None
        self.transformer = make_transformer()
        self.model_path = model_path
        self.device = torch.device(device)

    def load_model(self):
        print(cfg.NAME)
        # TODO:填写以项目为根目录的模型路径
        self.model = Network(cfg)
        # 读取模型权重
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        # 开启测试模式
        self.model.eval()

    def run_image(self, path):
        img = self.read_image(path)
        with torch.no_grad():
            img = img.to(self.device)
            feature, prediction = self.model(img)
            prediction = torch.argmax(prediction, dim=1)
        return feature, prediction

    def read_image(self, path):
        img = Image.open(path).convert("RGB")
        img = np.array(img)[:, :, [2, 1, 0]]
        # image = np.array(img)[:, :, [2, 1, 0]]
        img = self.transformer(img)
        # 只跑一张图像，添加第0维度（网络只支持4维的）
        img = torch.unsqueeze(img, 0)
        return img


if __name__ == '__main__':
    print(torch.__version__)
    parser = argparse.ArgumentParser(description="Classifier network")
    parser.add_argument(
        "--config-file",
        default="./config_file/resnet18.yaml",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    model = Classifier(cfg, device='cuda')
    # 加载模型
    model.load_model()
    # 返回的是GPU的tensor，可以用tolist()或item()取出
    # tolist()返回的是一个二维列表，取出[0]

    data_dir, image_lists, level_list = load_data_list()
    feature_list = []
    for i in range(100):
        feature, _ = model.run_image(os.path.join(data_dir, image_lists[i]))
        feature_list.append(feature.cpu().numpy())
        if i % 10 == 0:
            print(i)
    # image_path = './dataset/test/62_left.jpeg'
    # feature, prediction = model.run_image(image_path)
    # # 处理tensor
    # feature_list = feature.tolist()
    # feature512 = feature_list[0]
    # print('')
