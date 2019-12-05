#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.imagenet import ImageNetDataset
from model.data_parallel import get_data_parallel
from model.deeper_cnn import ImageNetDeeperCNN
from model.imagenet_cnn import ImageNetCNN
from model.pretrained_model import PretrainedModel
from running_log import RunningLog


def load_epoch(save_path: str, epoch: int) -> Any:
    print('loading from epoch.%04d.pth' % epoch)
    return torch.load(os.path.join(save_path, 'epoch.%04d.pth' % epoch),
                      map_location='cpu')


def eval_model(model: nn.Module, data_loader: DataLoader,
               device: torch.device) -> float:
    total_count, correct_count = 0, 0
    for data in tqdm(data_loader, desc='Eval'):
        data = [x.to(device) for x in data]
        total_count += data[0].size(0)
        output = model(data[0])
        correct_count += (torch.argmax(output, dim=1) == data[1]).sum().item()
    return correct_count / total_count


def main():
    parser = argparse.ArgumentParser()
    # Common Options
    parser.add_argument('--model', choices=['CNN', 'DeeperCNN', 'Pretrained'],
                        default='Pretrained', help='model to run')
    parser.add_argument('--task', choices=['train', 'valid', 'test'],
                        default='train', help='task to run')
    parser.add_argument('--dataset_path', help='path to the dataset folder',
                        default='data')
    parser.add_argument('--save_path', help='path for saving models and codes',
                        default='save/test')
    parser.add_argument('--gpu', type=lambda x: list(map(int, x.split(','))),
                        default=[], help="GPU ids separated by `,'")
    parser.add_argument('--load', type=int, default=0,
                        help='load module training at give epoch')
    parser.add_argument('--epoch', type=int, default=200, help='epoch to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--log_every_iter', type=int, default=100,
                        help='log loss every numbers of iteration')
    parser.add_argument('--valid_every_epoch', type=int, default=5,
                        help='run validation every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--save_every_epoch', type=int, default=10,
                        help='save model every numbers of epoch; '
                             '0 for disabling')
    parser.add_argument('--comment', default='', help='comment for tensorboard')
    # Pretrained Model Options
    parser.add_argument('--pretrained_model', choices=[
        'alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16',
        'vgg16_bn', 'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50',
        'resnet101', 'resnet152', 'squeezenet1_0', 'squeezenet1_1',
        'densenet121', 'densenet169', 'densenet161', 'densenet201',
        'inception_v3', 'googlenet', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
        'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'mobilenet_v2',
        'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2',
        'wide_resnet101_2', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
        'mnasnet1_3', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'SENet', 'senet154', 'se_resnet50', 'se_resnet101',
        'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d'],
        default='efficientnet-b3', help='pretrained model to use')
    # Build model
    args = parser.parse_args()
    running_log = RunningLog(args.save_path)
    running_log.set('parameters', vars(args))
    os.makedirs(args.save_path, exist_ok=True)
    if args.model == 'CNN':
        model = ImageNetCNN()
        train_transform = ImageNetCNN.get_transform()
        valid_transform = ImageNetCNN.get_transform()
    elif args.model == 'DeeperCNN':
        model = ImageNetDeeperCNN()
        train_transform = ImageNetDeeperCNN.get_transform()
        valid_transform = ImageNetDeeperCNN.get_transform()
    elif args.model == 'Pretrained':
        model = PretrainedModel(args.pretrained_model)
        train_transform = PretrainedModel.get_train_transform(
            args.pretrained_model)
        valid_transform = PretrainedModel.get_valid_transform(
            args.pretrained_model)
    else:
        raise RuntimeError('Unknown model')
    model: nn.Module = get_data_parallel(model, args.gpu)
    device = torch.device("cuda:%d" % args.gpu[0] if args.gpu else "cpu")
    optimizer_state_dict = None
    if args.load > 0:
        model_state_dict, optimizer_state_dict = \
            load_epoch(args.save_path, args.load)
        model.load_state_dict(model_state_dict)
    model.to(device)
    running_log.set('state', 'interrupted')
    # Start training or testing
    if args.task == 'train':
        model.train()
        train_dataset = ImageNetDataset(os.path.join(
            args.dataset_path, 'train.txt'), transform=train_transform)
        train_data_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True)
        valid_data_loader = None
        optimizer = Adam(model.parameters(), lr=args.learning_rate)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
        criterion = nn.CrossEntropyLoss().to(device)
        writer = SummaryWriter(comment=args.comment or
                                       os.path.basename(args.save_path))
        step = 0
        for epoch in tqdm(range(args.load + 1, args.epoch + 1), desc='Epoch'):
            losses = []
            for iter, data in enumerate(tqdm(train_data_loader, desc='Iter'),
                                        1):
                data = [x.to(device) for x in data]
                output = model(data[0])
                loss = criterion(output, data[1])
                losses.append(loss.item())
                writer.add_scalar('train/loss', loss.item(), step)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if iter % args.log_every_iter == 0:
                    # noinspection PyStringFormat
                    tqdm.write('epoch:[%d/%d] iter:[%d/%d] Loss=%.5f' %
                               (epoch, args.epoch, iter, len(train_data_loader),
                                np.mean(losses)))
                    losses = []
                step += 1
            if args.valid_every_epoch and epoch % args.valid_every_epoch == 0:
                if valid_data_loader is None:
                    valid_dataset = ImageNetDataset(
                        os.path.join(args.dataset_path, 'val.txt'),
                        transform=valid_transform)
                    valid_data_loader = DataLoader(valid_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=False)
                model.eval()
                acc = eval_model(model, valid_data_loader, device)
                tqdm.write('Accuracy=%f' % acc)
                writer.add_scalar('eval/acc', acc, epoch)
                model.train()
            if args.save_every_epoch and epoch % args.save_every_epoch == 0:
                tqdm.write('saving to epoch.%04d.pth' % epoch)
                torch.save((model.state_dict(), optimizer.state_dict()),
                           os.path.join(args.save_path,
                                        'epoch.%04d.pth' % epoch))
    elif args.task == 'valid':
        model.eval()
        valid_dataset = ImageNetDataset(os.path.join(
            args.dataset_path, 'val.txt'), transform=valid_transform)
        valid_data_loader = DataLoader(valid_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False)
        acc = eval_model(model, valid_data_loader, device)
        print('Accuracy=%f' % acc)
    elif args.task == 'test':
        model.eval()
        test_dataset = ImageNetDataset(
            os.path.join(args.dataset_path, 'test.txt'),
            is_test=True, transform=valid_transform)
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=False)
        predictions = []
        for data in tqdm(test_data_loader, desc='Test'):
            data = data.to(device)
            output = model(data)
            predictions += torch.argmax(output, dim=1).tolist()
        with open(os.path.join(args.save_path, 'predictions.txt'), 'w') as f:
            f.write('Id,Category\n')
            for image, prediction in zip(test_dataset.images, predictions):
                f.write('%s,%d\n' % (os.path.basename(image), prediction))
    running_log.set('state', 'succeeded')


if __name__ == '__main__':
    main()
