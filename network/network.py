import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from utils.focal_loss import FocalLoss
from . import resnet
from structures.image_list import to_image_list

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.resnet = resnet.Resnet(cfg)
        # 比例调整
        self.focal_loss = FocalLoss(5, torch.tensor(cfg.FOCAL_LOSS_ALPHA))
        # nn.init.kaiming_uniform_(self.fc2.weight, a=1)
        # nn.init.constant_(self.fc2.bias, 0)

    def forward(self, images, target=None):
        # TODO：model1
        # # 输入变为image_list
        feature, x = self.resnet(images)
        # x = self.fc2(feature)
        # feature = self.resnet(images)
        # x = self.relu1(feature)
        # x = self.fc2(x)

        if self.training:
            losses = {}
            # cross_entropy = F.cross_entropy(x, target)
            # losses.update(cross_entropy=cross_entropy)
            focal_loss = 2 * self.focal_loss(x, target)
            losses.update(focal_loss=focal_loss)
            return losses, x

        return feature, x
