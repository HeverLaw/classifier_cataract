from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import torch
import numpy as np
import random
import torch.nn.functional as F
import argparse

from config import cfg
from network import Network
from data import make_data_loader

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def test_net(cfg):
    NAME = cfg.NAME
    print(cfg.NAME)
    model_path = os.path.join(cfg.OUTPUT_DIR, 'final_model.pth')
    model = Network(cfg)
    device = torch.device(cfg.DEVICE)
    # 读取模型权重
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    dataloader_val = make_data_loader(cfg, is_train=False)
    inference(model, dataloader_val, device)


def inference(model, dataloader_val, device):
    global acc
    losses = []
    correct = [0] * 4
    total = [0] * 4
    acc = [0] * 4
    total_sum = 0
    correct_sum = 0
    model.eval()
    with torch.no_grad():
        for iteration, (images, levels) in enumerate(dataloader_val):
        # # TODO:测试other
        # for iteration, (images, levels, img_paths) in enumerate(dataloader_val):
            images = images.tensors
            images = images.to(device)
            levels_cuda = torch.tensor(levels).cuda()
            _, prediction = model(images)
            # 计算loss
            loss = F.cross_entropy(prediction, levels_cuda)
            losses.append(loss.item())
            # 获取分类结果，进行投票，统计结果
            prediction = torch.argmax(prediction, dim=1)
            res = prediction == levels_cuda
            res = res.tolist()
            # # TODO:测试other
            # img_paths = list(img_paths)
            for i in range(len(levels)):
                label_single = levels[i]
                if res[i]:
                    correct[label_single] += 1
                    # # TODO:测试other
                    # img_name_list[label_single].append(img_paths[i])
                total[label_single] += 1
            # print('iter:', iteration)
    # 数据统计/输出
    accuracy = sum(correct) / sum(total)
    acc_str = 'Acc_class: %f, ' % (accuracy)
    print('avg loss:', np.mean(losses))
    for acc_idx in range(4):
        try:
            acc[acc_idx] = correct[acc_idx] / total[acc_idx]
        except:
            acc[acc_idx] = 0
        finally:
            acc_str += 'class%d: %f, ' % (acc_idx + 1, acc[acc_idx])
    avg_acc = np.mean(acc)
    print(acc_str)
    # # TODO:测试other
    # return avg_acc, img_name_list
    return avg_acc, accuracy


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print(torch.__version__)
    # 设置随机种子
    setup_seed(2020)
    parser = argparse.ArgumentParser(description="Classifier network")
    parser.add_argument(
        "--config-file",
        default="./config_file/pretrained.yaml",
        metavar="FILE",
        help="path to config file",
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    test_net(cfg)
