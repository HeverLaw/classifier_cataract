from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import datetime
import torch
import numpy as np
import random
import argparse

from config import cfg
from network import Network
from data import make_data_loader
from solver import make_optimizer, make_lr_scheduler
from utils.metric_logger import MetricLogger
from tools.test_net import inference
from structures.image_list import to_image_list

# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def train_net(cfg):
    print(cfg.NAME)
    print(cfg.DESCRIPTION)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    model = Network(cfg)
    device = torch.device(cfg.DEVICE)
    if cfg.PRETRAINED:
        model_path = cfg.PRETRAIN_MODEL
        model.load_state_dict(torch.load(model_path))
        print('loding weight from ', model_path)
    model.to(device)
    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)
    # output_dir = cfg.OUTPUT_DIR
    data_loader = make_data_loader(cfg, is_train=True)
    args = {}
    args['iteration'] = 0
    do_train(model, data_loader, optimizer, scheduler, device, args, cfg)
    print(cfg.NAME)
    return model


def do_train(
    model,
    data_loader,
    optimizer,
    scheduler,
    device,
    args,
    cfg,
):
    meters = MetricLogger(delimiter="  ")
    start_iter = args["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()
    # 这里走了一个epoch，和自己规划的iter不一样
    iter_count = 0
    max_acc = max_acc_avg = 0
    data_loader_val = make_data_loader(cfg, is_train=False)
    for epoch in range(cfg.SOLVER.EPOCH):
        for iteration, (images, levels) in enumerate(data_loader, start_iter):
            # images = to_image_list(images)
            data_time = time.time() - end
            iter_count += 1
            scheduler.step()
            images = images.tensors
            images = images.to(device)
            levels = torch.tensor(levels).cuda()
            loss_dict, prediction = model(images, levels)
            # 计算accuracy
            prediction = torch.argmax(prediction, dim=1)
            res = prediction == levels
            res = res.tolist()
            correct = 0
            for i in range(len(res)):
                if res[i]:
                    correct += 1
            train_acc = correct / len(res)

            losses = sum(loss for loss in loss_dict.values())
            meters.update(loss=losses, train_acc=train_acc, **loss_dict)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iter_count % 10 == 0:
                print("epoch: {epoch}, iter: {iter}, {meters}, lr: {lr:.6f}".format(
                    epoch=epoch, iter=iter_count, meters=str(meters), lr=optimizer.param_groups[0]["lr"]))
        if epoch >= 0 and epoch % 2 == 1:
            # 训练时测试，遇到效果好的保存模型，先测试再打开train，隔两次测试一次
            # TODO:测试other
            print('epoch ', epoch, ': testing the model')
            acc_avg, accuracy = inference(model, data_loader_val, device)
            if accuracy > max_acc:
                max_acc = accuracy
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'model' + str(epoch) + '.pth'))
                print("saving model")
            elif acc_avg > max_acc_avg:
                max_acc_avg = acc_avg
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'model' + str(epoch) + '.pth'))
                print("saving model")
            model.train()

    # 计算时间
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    print("Total training time: {} ({:.4f} s / epoch)".format(
            total_time_str, total_training_time / cfg.SOLVER.EPOCH))
    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, 'final_model.pth'))
    return model


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
    # parser.add_argument(
    #     "--description",
    #     default="describe the project",
    # )
    args = parser.parse_args()
    # print(args.description)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    model = train_net(cfg)



