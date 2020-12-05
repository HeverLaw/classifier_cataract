# -*- coding: utf-8 -*-
import os
import torch
import argparse
import pandas as pd
import base64
import re
from io import BytesIO
from config import cfg
from tools.predictor_resize import Classifier
from PIL import Image


# 测试工具
def load_data_list():
    '''
    用于加载测试数据
    :return:
    '''
    csv_dir = './dataset'
    data_dir = './dataset/resized_train_cropped/resized_train_cropped'
    filename = os.path.join(csv_dir, '32000test.csv')
    df = pd.read_csv(filename)
    image_list = list(df['image'])
    level_list = list(df['level'])
    return data_dir, image_list, level_list


# 测试工具
def image_to_base64(image_path):
    img = Image.open(image_path).convert('RGB')
    output_buffer = BytesIO()
    img.save(output_buffer, format='JPEG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


# 测试工具
def base64_to_image(base64_str, image_path=None):
    base64_str = base64_str.decode('utf-8')
    base64_data = re.sub('^data:image/.+;base64,', '', base64_str)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    if image_path:
        img.save(image_path)
    return img


def load_config():
    '''
    加载配置文件
    :return: cfg配置文件对象
    '''
    print(torch.__version__)
    parser = argparse.ArgumentParser(description="Classifier network")
    parser.add_argument(
        "--config-file",
        default="./config_file/resnet34_32000_freeze.yaml",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    return cfg


# 配置模型路径
model_path = {
    'resnet18': './models/resnet18.pth',
    'resnet34': './models/resnet34.pth',
}


if __name__ == '__main__':
    # 加载测试数据
    data_dir, image_lists, level_list = load_data_list()

    # 加载配置信息
    cfg = load_config()

    # 模型存起来，等待接收数据，收到请求调用model.run_image即可
    model = Classifier(cfg, model_path, device='cuda')

    # 加载模型
    model.load_model()

    # 模拟接收数据
    path = os.path.join(data_dir, image_lists[0])
    # 假设接收到base64的图像
    base64_image = image_to_base64(path)
    # 再转化为PIL.Image.Image对象
    img = base64_to_image(base64_image)

    # 方式一：直接传入PIL.Image.Image对象
    prediction, score = model.run_image(img=img)
    # 输出分类结果
    print(prediction, score)

    # 方式二：使用路径读取图像
    prediction, score = model.run_image(path=os.path.join(data_dir, image_lists[1]))
    # 输出分类结果
    print(prediction, score)
