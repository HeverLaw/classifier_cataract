import torch.utils.data
import os
import pandas as pd
from data.datasets import ListDataset
from data.datasets import ListDatasetVal
from .transforms import build_transforms
from .collate_batch import BatchCollator
from .sampler import IterationBasedBatchSampler


def build_dataset(transforms, is_train=True, dataset_name=None):
    '''
    生成数据集列表，生成的dataloader设置如何读取数据
    :param transforms:
    :param is_train:
    :return:
    '''
    data_dir = './dataset'
    filename = os.path.join(data_dir, 'train.csv')
    # 选择数据集
    if is_train:
        if dataset_name is not None:
            if dataset_name == 'train':
                filename = os.path.join(data_dir, 'train.csv')
                img_dir = os.path.join(data_dir, 'image')
            else:
                raise AssertionError("dataset is not existed!")
        else:
            filename = os.path.join(data_dir, 'train.csv')
            img_dir = os.path.join(data_dir, 'image')
    else:
        if dataset_name is not None:
            if dataset_name == 'train':
                filename = os.path.join(data_dir, 'train.csv')
                img_dir = os.path.join(data_dir, 'image')
            elif dataset_name == 'val':
                filename = os.path.join(data_dir, 'val.csv')
                img_dir = os.path.join(data_dir, 'image')
            else:
                raise AssertionError("dataset is not existed!")
        else:
            filename = os.path.join(data_dir, 'val.csv')
            img_dir = os.path.join(data_dir, 'image')
    print('dataset file is ', filename)
    df = pd.read_csv(filename)
    image_lists = list(df['image'])
    level_list = list(df['level'])
    dataset = ListDataset(img_dir, image_lists, level_list, transforms)

    return dataset


def make_data_sampler(dataset, shuffle):
    '''
    生成一个sampler，设置shuffle
    :param dataset:
    :param shuffle:
    :return:
    '''
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(sampler, images_per_batch, start_iter=0, num_iters=None):
    '''
    生成一个batch sampler，可以用于获取多张图像，batch_size，简单版可以直接在dataloader里面设置batch_size
    :param sampler:
    :param images_per_batch:
    :return:
    '''
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )

    return batch_sampler


def make_data_loader(cfg, is_train=True):
    '''
    创建dataloader，此前需要读取data/transform/创建dataset
    :param cfg:
    :param is_train:
    :return:
    '''
    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    transforms = build_transforms(cfg, is_train)
    dataset = build_dataset(transforms, is_train,
                            cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST)
    # # TODO:测试other
    # dataset = build_other_dataset(transforms, is_train)
    shuffle = False
    if is_train:
        shuffle = True
    # num_iters = cfg.SOLVER.MAX_ITER
    sampler = make_data_sampler(dataset, shuffle)
    # batch_sampler = make_batch_data_sampler(sampler, images_per_batch, num_iters=num_iters)
    batch_sampler = make_batch_data_sampler(sampler, images_per_batch)
    collator = BatchCollator()
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=4,
                                              collate_fn=collator)
    # data_loader = torch.utils.data.DataLoader(dataset,
    #                                           batch_size=images_per_batch,
    #                                           shuffle=shuffle,
    #                                           num_workers=4, )
    return data_loader
