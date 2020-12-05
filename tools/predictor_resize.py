from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import numpy as np
import random
import argparse
import pandas as pd
import time
from config import cfg
from network import Network
from torchvision import transforms
import PIL
from PIL import Image
from torchvision.transforms import functional as F
from torch.nn.functional import softmax
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc


# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


class Resize(object):
    def __init__(self, min_size, max_size):
        '''
        数据转换的工具，用于保持比例resize
        :param min_size:
        :param max_size:
        '''
        if not isinstance(min_size, (list, tuple)):
            min_size = (min_size,)
        self.min_size = min_size
        self.max_size = max_size

    def get_size(self, image_size):
        w, h = image_size
        size = random.choice(self.min_size)
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        # image = F.resize(image, (512,512))
        image = F.resize(image, size)
        return image


def make_transformer(cfg):
    '''
    创造一个transformer，用于数据预处理
    :param cfg:
    :return:
    '''
    transformer = transforms.Compose(
        [
            transforms.ToPILImage(),
            Resize(cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ]
    )
    return transformer


class Classifier:
    def __init__(self, config, model_path, device='cpu'):
        '''
        使用resnet34或resnet18，直接使用库里面的transforms
        :param config: config对象
        :param model_path: 模型的路径
        :param device: 使用cpu或者cuda
        '''
        super(Classifier, self).__init__()
        self.cfg = config
        self.model = None
        self.transformer = make_transformer(cfg)
        self.model_path = model_path[cfg.BACKBONE]
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

    def run_image(self, path=None, img=None, is_test=False):
        # 接收图像
        assert path is not None or img is not None, '至少传入path或者PIL.Image格式的图像'
        if path is not None:
            img = Image.open(path).convert("RGB")
            img = self.pre_process_image(img)
        else:
            assert isinstance(img, PIL.Image.Image), '输入图像需要为Image.Image的RGB格式'
            img = self.pre_process_image(img)
        ##############################################
        with torch.no_grad():
            img = img.to(self.device)
            feature, prediction = self.model(img)
            score_list = softmax(prediction, dim=1).cpu().numpy()
            prediction = torch.argmax(prediction, dim=1).item()
        if is_test:
            return prediction, score_list
        else:
            return prediction, score_list[0][prediction]


    def pre_process_image(self, img):
        '''
        输入图像需要为
        :param path:
        :return:
        '''
        img = np.array(img)
        img = self.transformer(img)
        # 只跑一张图像，添加第0维度（网络只支持4维的）
        img = torch.unsqueeze(img, 0)
        return img


model_path = {
    'resnet18': './models/resnet18.pth',
    'resnet34': './models/resnet34_best.pth',
}


if __name__ == '__main__':
    # ----------------------读取配置----------------
    print(torch.__version__)
    parser = argparse.ArgumentParser(description="Classifier network")
    parser.add_argument(
        "--config-file",
        default="./config_file/resnet34_fine_tune.yaml",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    # ---------------------加载模型----------------
    model = Classifier(cfg, model_path=model_path, device='cuda')
    model.load_model()
    # 返回的是GPU的tensor，可以用tolist()或item()取出
    # tolist()返回的是一个二维列表，取出[0]

    # --------------------读取数据-------------------
    def load_data_list():
        csv_dir = './dataset'
        data_dir = './dataset/image'
        filename = os.path.join(csv_dir, 'val.csv')
        df = pd.read_csv(filename)
        image_list = list(df['image'])
        level_list = list(df['level'])
        return data_dir, image_list, level_list
    data_dir, image_lists, level_list = load_data_list()

    # -------------------------测试与评价-----------
    level_array = np.array(level_list).reshape(-1, 1)
    level_one_hot = OneHotEncoder(sparse=False).fit_transform(level_array)
    res = [0] * 4
    right_sum = 0
    sum_time = 0
    score_lists = np.zeros((1, 4))
    for i in range(len(image_lists)):
        start_time = time.time()
        prediction, score_list = model.run_image(os.path.join(data_dir, image_lists[i]), is_test=True)
        if i == 0:
            score_lists = score_list
        else:
            score_lists = np.append(score_lists, score_list, axis=0)
        end_time = time.time()
        sum_time += (end_time - start_time)
        if i % 10 == 0:
            print(i)
        if prediction == level_list[i]:
            res[prediction] += 1
            right_sum += 1
    print('avg time:', sum_time / len(image_lists))

    # -----------------计算准确率-----------------
    lres = [0] * 4
    sum_cnt = 0
    for i in range(len(level_list)):
        lres[level_list[i]] += 1
        sum_cnt += 1
    ratio = [0] * 4
    acc_str = 'Accuracy: %f, ' % (sum(res) / sum_cnt)
    for i in range(len(res)):
        ratio[i] = res[i] / lres[i]
        acc_str += 'class%d: %f, ' % (i + 1, ratio[i])
    print(acc_str)

    # --------------计算roc-auc-----------------
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_sum = 0
    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(level_one_hot[:, i], score_lists[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        auc_sum += roc_auc[i]
    auc_avg = auc_sum / 4

    # ------------微平均--------------------------
    fpr["micro"], tpr["micro"], _ = roc_curve(level_one_hot.ravel(), score_lists.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # -------------宏平均-----------------------
    # 首先汇总所有FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))
    # 然后再用这些点对ROC曲线进行插值
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 4
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # -------------输出信息-----------------------
    print('roc_auc:', roc_auc)
